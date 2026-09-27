#!/usr/bin/env python3
"""
generate_samples.py — Genera outputs/sample_queries.json con preguntas reales.

Corre 4 preguntas de ejemplo contra el sistema RAG ya indexado. Guarda:
  - outputs/sample_queries.json      -> array con SOLO las 3 claves exactas
                                         que pide la consigna (user_question,
                                         system_answer, chunks_related), sin
                                         ningún campo extra.
  - outputs/sample_evaluations.json  -> (bonus) el resultado del agente
                                         evaluador para cada una de esas
                                         mismas preguntas, como archivo aparte.

Uso (después de correr build_index.py y tener OPENAI_API_KEY seteada):
    python src/generate_samples.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from query import run_with_evaluation  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
QUERIES_PATH = OUTPUT_DIR / "sample_queries.json"
EVALUATIONS_PATH = OUTPUT_DIR / "sample_evaluations.json"

SAMPLE_QUESTIONS = [
    "¿Cuántos días de vacaciones tengo por año y cómo los solicito?",
    "¿Qué pasa si pierdo el acceso a mi dispositivo de autenticación de dos factores?",
    "¿Cómo funciona el bono de referidos y cuánto paga?",
    "¿En qué plazo tengo que reclamar si mi sueldo se liquidó mal?",
]


def main():
    query_results = []
    evaluations = []

    for i, question in enumerate(SAMPLE_QUESTIONS, start=1):
        print(f"[{i}/{len(SAMPLE_QUESTIONS)}] Consultando: {question}")
        result, evaluation = run_with_evaluation(question)
        query_results.append(result)
        evaluations.append({"user_question": question, "evaluation": evaluation})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    QUERIES_PATH.write_text(json.dumps(query_results, ensure_ascii=False, indent=2), encoding="utf-8")
    EVALUATIONS_PATH.write_text(json.dumps(evaluations, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nGuardado: {QUERIES_PATH}")
    print(f"Guardado (bonus): {EVALUATIONS_PATH}")


if __name__ == "__main__":
    main()