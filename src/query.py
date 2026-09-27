#!/usr/bin/env python3
"""
query.py — Pipeline de consulta (online, se corre por cada pregunta).

Uso:
    python src/query.py "¿Cuántos días de vacaciones tengo por año?"
    python src/query.py "pregunta" --top-k 5 --no-evaluate

Etapas del pipeline RAG (cada una es una función separada, ver rúbrica):
    1. load_index()          — carga chunks + embeddings ya generados
    2. embed_query()         — convierte la pregunta a un vector
    3. search_similar_chunks() — búsqueda vectorial (k-NN exacto, similitud coseno)
    4. build_context_prompt() — ensambla los chunks recuperados en el prompt
    5. generate_answer()     — el LLM genera la respuesta usando ese contexto
    6. format_output()       — arma el JSON con user_question/system_answer/chunks_related
    (bonus) evaluate_answer() — un segundo LLM puntúa la calidad de la respuesta
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_utils import with_retries  # noqa: E402
from evaluator import evaluate_answer  # noqa: E402

INDEX_PATH = Path(__file__).resolve().parent.parent / "index" / "faq_index.json"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
DEFAULT_TOP_K = 3


def load_index(path: Path = INDEX_PATH) -> tuple[list[dict], np.ndarray]:
    """
    Carga el índice generado por build_index.py. Devuelve la lista de chunks
    (con su texto e id) y una matriz numpy con todos los embeddings, para
    poder vectorizar la búsqueda de similitud.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el índice en {path}. Corré primero: python src/build_index.py"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = data["chunks"]
    embeddings_matrix = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    return chunks, embeddings_matrix


@with_retries()
def embed_query(question: str, model: str = EMBEDDING_MODEL) -> np.ndarray:
    """Convierte la pregunta del usuario en un embedding (mismo modelo que los chunks)."""
    from openai import OpenAI

    client = OpenAI()
    response = client.embeddings.create(model=model, input=[question])
    return np.array(response.data[0].embedding, dtype=np.float32)


def cosine_similarity_batch(query_vector: np.ndarray, chunk_matrix: np.ndarray) -> np.ndarray:
    """
    Similitud coseno entre el vector de la pregunta y TODOS los vectores de
    chunks a la vez (vectorizado con numpy). Fórmula: cos(a, b) = (a·b) / (|a| |b|)

    Búsqueda vectorial elegida: k-NN exacto (fuerza bruta) sobre similitud
    coseno. Con ~26 chunks esto es prácticamente instantáneo y evita sumar
    una dependencia extra (FAISS/Annoy) que complicaría la reproducibilidad
    del proyecto para un corpus de este tamaño.
    """
    query_norm = query_vector / np.linalg.norm(query_vector)
    chunk_norms = chunk_matrix / np.linalg.norm(chunk_matrix, axis=1, keepdims=True)
    return chunk_norms @ query_norm


def search_similar_chunks(
    query_vector: np.ndarray,
    chunks: list[dict],
    embeddings_matrix: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    Búsqueda k-NN: devuelve los `top_k` chunks más similares a la pregunta,
    ordenados de mayor a menor similitud, cada uno con su score.
    """
    similarities = cosine_similarity_batch(query_vector, embeddings_matrix)
    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [
        {
            "id": chunks[i]["id"],
            "text": chunks[i]["text"],
            "similarity": round(float(similarities[i]), 4),
        }
        for i in top_indices
    ]


def build_context_prompt(question: str, chunks_related: list[dict]) -> list[dict]:
    """Ensambla los chunks recuperados como contexto dentro del prompt para el LLM."""
    context_text = "\n\n".join(
        f"[Fragmento {c['id']}]\n{c['text']}" for c in chunks_related
    )
    system_prompt = (
        "Sos un asistente de soporte de FAQs para NimbusHR, una plataforma de "
        "recursos humanos. Respondé la pregunta del usuario ÚNICAMENTE con la "
        "información de los fragmentos de contexto que te paso a continuación. "
        "Si el contexto no alcanza para responder con certeza, decilo "
        "explícitamente en vez de inventar una respuesta. Sé conciso (máximo "
        "3-4 oraciones) y no menciones que estás usando 'fragmentos' o "
        "'contexto' en la respuesta, respondé de forma natural como lo haría "
        "un agente de soporte.\n\n"
        f"CONTEXTO:\n{context_text}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


@with_retries()
def generate_answer(question: str, chunks_related: list[dict], model: str = CHAT_MODEL) -> str:
    """Le pide al LLM que genere la respuesta final, basada en los chunks recuperados (RAG)."""
    from openai import OpenAI

    client = OpenAI()
    messages = build_context_prompt(question, chunks_related)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def format_output(question: str, answer: str, chunks_related: list[dict]) -> dict:
    """Arma el JSON de salida con las 3 claves exactas que pide la consigna."""
    return {
        "user_question": question,
        "system_answer": answer,
        "chunks_related": [
            {"id": c["id"], "text": c["text"], "similarity": c["similarity"]}
            for c in chunks_related
        ],
    }


def run(question: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """
    Orquesta el pipeline de consulta principal. Devuelve el JSON con
    EXACTAMENTE las 3 claves que pide la consigna (user_question,
    system_answer, chunks_related) — sin campos extra, para no romper la
    validación estricta del contrato de salida.
    """
    chunks, embeddings_matrix = load_index()

    query_vector = embed_query(question)
    chunks_related = search_similar_chunks(query_vector, chunks, embeddings_matrix, top_k)
    answer = generate_answer(question, chunks_related)

    return format_output(question, answer, chunks_related)


def run_with_evaluation(question: str, top_k: int = DEFAULT_TOP_K) -> tuple[dict, dict]:
    """
    Igual que `run()`, pero además corre el agente evaluador (bonus) sobre
    el resultado. Devuelve una TUPLA (result, evaluation) en vez de mezclar
    la evaluación dentro del mismo diccionario — así el JSON del pipeline
    principal nunca deja de tener exactamente 3 claves; la evaluación viaja
    siempre como un objeto aparte.
    """
    result = run(question, top_k=top_k)
    evaluation = evaluate_answer(
        result["user_question"], result["system_answer"], result["chunks_related"]
    )
    return result, evaluation


def main():
    parser = argparse.ArgumentParser(description="Pipeline de consulta RAG para FAQs de NimbusHR")
    parser.add_argument("question", help="Pregunta del usuario, entre comillas")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Cantidad de chunks a recuperar")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Corre además el agente evaluador (bonus) e imprime su resultado por separado",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print(json.dumps({"error": "OPENAI_API_KEY no está seteada."}, ensure_ascii=False))
        sys.exit(1)

    if args.evaluate:
        result, evaluation = run_with_evaluation(args.question, top_k=args.top_k)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("\n--- Evaluación (bonus, objeto separado del JSON principal) ---")
        print(json.dumps(evaluation, ensure_ascii=False, indent=2))
    else:
        result = run(args.question, top_k=args.top_k)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()