"""
evaluator.py — Agente evaluador (bonus).

Recibe la pregunta, la respuesta generada y los chunks usados, y le pide a
un LLM (separado de la llamada que generó la respuesta) que la evalúe en
dos dimensiones: fidelidad al contexto (¿la respuesta se basa realmente en
los chunks, o inventa/alucina?) y completitud (¿responde toda la pregunta?).
Devuelve un score de 0 a 10 y una justificación.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_utils import with_retries  # noqa: E402

EVAL_MODEL = os.getenv("EVAL_MODEL", "gpt-4o-mini")

EVALUATOR_PROMPT = """Sos un agente evaluador de calidad para un sistema RAG de FAQs. \
Tu trabajo es puntuar qué tan buena es una respuesta generada, comparándola \
ÚNICAMENTE con los fragmentos de contexto que se usaron para generarla — no \
con tu propio conocimiento general.

Evaluá en base a estas dimensiones:
1. Fidelidad/relevancia: ¿la respuesta usa información que efectivamente está \
en los chunks recuperados, o inventa/alucina datos que no aparecen ahí?
2. Completitud: ¿la respuesta cubre lo que la pregunta pedía, o se queda corta?

Devolvé SOLO un JSON válido, sin texto adicional, con exactamente estos campos:
{"score": <entero de 0 a 10>, "reason": "<justificación de al menos 50 caracteres, mencionando observaciones concretas sobre fidelidad y completitud>"}
"""


def build_evaluation_prompt(question: str, answer: str, chunks_related: list[dict]) -> list[dict]:
    """Arma el prompt que le muestra al evaluador la pregunta, la respuesta y los chunks."""
    context_text = "\n\n".join(
        f"[Fragmento {c['id']}]\n{c['text']}" for c in chunks_related
    )
    user_content = (
        f"PREGUNTA DEL USUARIO:\n{question}\n\n"
        f"RESPUESTA GENERADA POR EL SISTEMA:\n{answer}\n\n"
        f"CHUNKS RECUPERADOS Y USADOS COMO CONTEXTO:\n{context_text}"
    )
    return [
        {"role": "system", "content": EVALUATOR_PROMPT},
        {"role": "user", "content": user_content},
    ]


@with_retries()
def _call_evaluator_model(messages: list[dict], model: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,  # el evaluador tiene que ser lo más determinístico posible
        max_tokens=200,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def evaluate_answer(question: str, answer: str, chunks_related: list[dict], model: str = EVAL_MODEL) -> dict:
    """
    Evalúa la respuesta generada. Devuelve {"score": int, "reason": str}.
    Ante cualquier falla (JSON inválido, error de API tras los reintentos),
    devuelve un resultado de fallback en vez de tirar abajo todo el pipeline
    de consulta — la evaluación es un valor agregado, no debe romper la
    respuesta principal si falla.
    """
    messages = build_evaluation_prompt(question, answer, chunks_related)
    try:
        raw_content = _call_evaluator_model(messages, model)
        data = json.loads(raw_content)
        score = int(data["score"])
        reason = str(data["reason"])
        if not (0 <= score <= 10) or len(reason) < 50:
            raise ValueError("El evaluador no respetó el contrato esperado.")
        return {"score": score, "reason": reason}
    except Exception as e:
        return {
            "score": 0,
            "reason": (
                "No se pudo completar la evaluación automática por un error "
                f"técnico ({type(e).__name__}). Este score no refleja la "
                "calidad real de la respuesta, solo indica una falla del "
                "agente evaluador."
            ),
        }
