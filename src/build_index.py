#!/usr/bin/env python3
"""
build_index.py — Pipeline de indexación (offline).

Se corre UNA vez (o cada vez que cambia data/faq_document.txt), y arma la
base de conocimiento consultable: divide el documento en chunks, genera un
embedding por chunk, y guarda todo en index/faq_index.json.

Uso:
    python src/build_index.py

Etapas del pipeline (cada una es una función separada, ver rúbrica):
    1. load_document()          — carga el documento de texto plano
    2. chunk_document()         — lo divide en chunks por oraciones/párrafos
    3. generate_embeddings()    — genera un vector por chunk (OpenAI)
    4. save_index()             — persiste chunks + embeddings en JSON
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_utils import with_retries  # noqa: E402

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "faq_document.txt"
INDEX_PATH = Path(__file__).resolve().parent.parent / "index" / "faq_index.json"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# --- Parámetros de chunking por oraciones/párrafos -------------------------
# Estrategia elegida: agrupar oraciones consecutivas (respetando los cortes
# de párrafo del documento original) hasta alcanzar TARGET_CHUNK_TOKENS,
# sin superar nunca MAX_CHUNK_TOKENS ni quedar por debajo de MIN_CHUNK_TOKENS
# (salvo el último chunk del documento, que puede ser más corto).
MIN_CHUNK_TOKENS = 50
TARGET_CHUNK_TOKENS = 150
MAX_CHUNK_TOKENS = 500


def load_document(path: Path = DATA_PATH) -> str:
    """Carga el documento de texto plano, manejando explícitamente la codificación."""
    return path.read_text(encoding="utf-8")


_encoding = None
_tiktoken_available = None


def count_tokens(text: str) -> int:
    """
    Cuenta tokens con el tokenizer de OpenAI (tiktoken), para que los límites
    de chunk_size sean consistentes con lo que el modelo realmente va a ver.

    Si tiktoken no puede inicializarse (por ejemplo, sin acceso a internet
    la primera vez que necesita bajar su tabla de tokens), se usa una
    aproximación basada en caracteres (~4 caracteres por token, heurística
    estándar para inglés/español) para que el pipeline no se caiga.
    """
    global _encoding, _tiktoken_available

    if _tiktoken_available is None:
        try:
            import tiktoken

            _encoding = tiktoken.get_encoding("cl100k_base")
            _tiktoken_available = True
        except Exception:
            _tiktoken_available = False

    if _tiktoken_available:
        return len(_encoding.encode(text))
    return max(1, len(text) // 4)


def split_into_paragraphs(text: str) -> list[str]:
    """Divide el documento en párrafos (separados por línea en blanco)."""
    paragraphs = [p.strip() for p in text.split("\n\n")]
    return [p for p in paragraphs if p]


def split_into_sentences(paragraph: str) -> list[str]:
    """
    Divide un párrafo en oraciones. Usa un regex simple que corta después de
    '.', '?' o '!' seguido de espacio y mayúscula — suficiente para el texto
    de FAQ que no tiene abreviaturas ambiguas complejas.
    """
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡])", paragraph)
    return [s.strip() for s in sentences if s.strip()]


def chunk_document(
    text: str,
    min_tokens: int = MIN_CHUNK_TOKENS,
    target_tokens: int = TARGET_CHUNK_TOKENS,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[str]:
    """
    Chunking por oraciones/párrafos: agrupa oraciones consecutivas hasta
    acercarse a `target_tokens`, sin cortar una oración a la mitad y sin
    mezclar contenido de dos párrafos distintos dentro del mismo chunk
    (así cada chunk preserva una unidad temática completa). Al final,
    fusiona los chunks que quedaron demasiado cortos (ver `merge_short_chunks`)
    para que todos respeten el rango [min_tokens, max_tokens].
    """
    raw_chunks: list[str] = []

    for paragraph in split_into_paragraphs(text):
        sentences = split_into_sentences(paragraph)
        current_sentences: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = count_tokens(sentence)

            # Si agregar esta oración se pasa del máximo, o ya llegamos al
            # target y agregar más la alejaría demasiado, cerramos el chunk.
            would_exceed_target = current_tokens + sentence_tokens > target_tokens
            has_min_content = current_tokens >= min_tokens

            if current_sentences and would_exceed_target and has_min_content:
                raw_chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_tokens = 0

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

            # Salvaguarda: si una sola oración ya es enorme, cerramos igual.
            if current_tokens > max_tokens:
                raw_chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_tokens = 0

        if current_sentences:
            raw_chunks.append(" ".join(current_sentences))

    return merge_short_chunks(raw_chunks, min_tokens, max_tokens)


def merge_short_chunks(
    chunks: list[str], min_tokens: int = MIN_CHUNK_TOKENS, max_tokens: int = MAX_CHUNK_TOKENS
) -> list[str]:
    """
    Fusiona chunks consecutivos que quedaron por debajo de `min_tokens`
    (por ejemplo, un título suelto o un párrafo muy corto) con el siguiente,
    mientras no se supere `max_tokens`. Al final, si el último chunk sigue
    corto, se fusiona hacia atrás con el anterior.
    """
    merged: list[str] = []
    i = 0
    while i < len(chunks):
        current = chunks[i]
        current_tokens = count_tokens(current)

        while current_tokens < min_tokens and i + 1 < len(chunks):
            candidate = f"{current} {chunks[i + 1]}"
            candidate_tokens = count_tokens(candidate)
            if candidate_tokens > max_tokens:
                break
            current = candidate
            current_tokens = candidate_tokens
            i += 1

        merged.append(current)
        i += 1

    if len(merged) >= 2 and count_tokens(merged[-1]) < min_tokens:
        combined = f"{merged[-2]} {merged[-1]}"
        if count_tokens(combined) <= max_tokens:
            merged[-2] = combined
            merged.pop()

    return merged


@with_retries()
def _embed_batch(client, texts: list[str], model: str) -> list[list[float]]:
    """Una sola llamada a la API de embeddings para un lote de textos."""
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def generate_embeddings(
    chunks: list[str], model: str = EMBEDDING_MODEL, batch_size: int = 16
) -> list[list[float]]:
    """
    Genera un embedding por chunk, llamando a la API en lotes (más rápido y
    más barato que uno por uno). Devuelve la lista de vectores en el mismo
    orden que los chunks de entrada.
    """
    from openai import OpenAI

    client = OpenAI()
    embeddings: list[list[float]] = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_embeddings = _embed_batch(client, batch, model)
        embeddings.extend(batch_embeddings)

    return embeddings


def save_index(
    chunks: list[str],
    embeddings: list[list[float]],
    model: str,
    path: Path = INDEX_PATH,
) -> None:
    """Persiste chunks + embeddings + metadata en un archivo JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "embedding_model": model,
        "chunk_count": len(chunks),
        "chunks": [
            {"id": i, "text": chunk, "embedding": embedding, "token_count": count_tokens(chunk)}
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ],
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def main():
    print("1/4 — Cargando documento...")
    document_text = load_document()

    print("2/4 — Dividiendo en chunks (por oraciones/párrafos)...")
    chunks = chunk_document(document_text)
    print(f"     -> {len(chunks)} chunks generados.")

    token_counts = [count_tokens(c) for c in chunks]
    print(
        f"     -> tokens por chunk: min={min(token_counts)}, "
        f"max={max(token_counts)}, promedio={sum(token_counts) / len(token_counts):.0f}"
    )

    print(f"3/4 — Generando embeddings con {EMBEDDING_MODEL}...")
    embeddings = generate_embeddings(chunks)
    print(f"     -> {len(embeddings)} embeddings generados.")

    print("4/4 — Guardando índice...")
    save_index(chunks, embeddings, EMBEDDING_MODEL)
    print(f"     -> Índice guardado en {INDEX_PATH}")


if __name__ == "__main__":
    main()
