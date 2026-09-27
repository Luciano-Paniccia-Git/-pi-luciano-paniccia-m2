"""
tests/test_core.py

Cómo correrlos:
    pip install -r requirements.txt
    pytest tests/test_core.py -v

No requieren OPENAI_API_KEY: testean la lógica de chunking, el cálculo de
similitud coseno y la construcción del JSON de salida, sin llamar a la API.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from build_index import (  # noqa: E402
    chunk_document,
    count_tokens,
    merge_short_chunks,
    split_into_paragraphs,
    split_into_sentences,
)
from query import cosine_similarity_batch, format_output, search_similar_chunks  # noqa: E402


# --- Chunking ------------------------------------------------------------

def test_split_into_paragraphs_splits_on_blank_lines():
    text = "Primer párrafo.\n\nSegundo párrafo con dos oraciones. Va acá."
    paragraphs = split_into_paragraphs(text)
    assert len(paragraphs) == 2
    assert paragraphs[0] == "Primer párrafo."


def test_split_into_sentences_splits_correctly():
    paragraph = "Esta es la primera oración. Esta es la segunda. ¿Y esta es una pregunta? Sí."
    sentences = split_into_sentences(paragraph)
    assert len(sentences) == 4


def test_chunk_document_respects_token_bounds():
    document = Path(__file__).resolve().parent.parent / "data" / "faq_document.txt"
    text = document.read_text(encoding="utf-8")
    chunks = chunk_document(text)

    assert len(chunks) >= 20, "La consigna pide 20+ chunks"

    for chunk in chunks:
        tokens = count_tokens(chunk)
        assert 50 <= tokens <= 500, f"Chunk fuera de rango: {tokens} tokens"


def test_merge_short_chunks_merges_below_minimum():
    short_chunks = ["Uno.", "Dos.", "Este es un chunk bastante más largo que supera el mínimo de tokens requerido para no ser fusionado con otros chunks vecinos en el proceso de fusión automática."]
    merged = merge_short_chunks(short_chunks, min_tokens=10, max_tokens=200)
    # los dos primeros, muy cortos, deberían fusionarse entre sí o con el siguiente
    assert len(merged) < len(short_chunks)


# --- Similitud coseno / búsqueda k-NN -------------------------------------

def test_cosine_similarity_identical_vectors_is_one():
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    matrix = np.array([vector], dtype=np.float32)
    similarities = cosine_similarity_batch(vector, matrix)
    assert abs(similarities[0] - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_vectors_is_zero():
    query = np.array([1.0, 0.0], dtype=np.float32)
    matrix = np.array([[0.0, 1.0]], dtype=np.float32)
    similarities = cosine_similarity_batch(query, matrix)
    assert abs(similarities[0] - 0.0) < 1e-6


def test_search_similar_chunks_returns_top_k_sorted_by_similarity():
    chunks = [
        {"id": 0, "text": "chunk cero"},
        {"id": 1, "text": "chunk uno"},
        {"id": 2, "text": "chunk dos"},
    ]
    embeddings_matrix = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float32
    )
    query_vector = np.array([1.0, 0.0], dtype=np.float32)

    results = search_similar_chunks(query_vector, chunks, embeddings_matrix, top_k=2)

    assert len(results) == 2
    assert results[0]["id"] == 0  # el más parecido al query
    assert results[0]["similarity"] >= results[1]["similarity"]


# --- Contrato de salida JSON ----------------------------------------------

def test_format_output_has_exact_required_keys():
    chunks_related = [{"id": 0, "text": "algún texto", "similarity": 0.87}]
    result = format_output("¿pregunta?", "una respuesta", chunks_related)

    assert set(result.keys()) == {"user_question", "system_answer", "chunks_related"}
    assert result["user_question"] == "¿pregunta?"
    assert result["system_answer"] == "una respuesta"
    assert result["chunks_related"][0]["id"] == 0
