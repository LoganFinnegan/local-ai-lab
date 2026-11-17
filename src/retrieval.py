from pathlib import Path

import numpy as np
from fastembed import TextEmbedding


EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_text(file_path):
    path = Path(file_path)

    return path.read_text(encoding="utf-8")


def chunk_text(text):
    paragraphs = text.split("\n\n")

    chunks = []

    for paragraph in paragraphs:
        cleaned_paragraph = paragraph.strip()

        if cleaned_paragraph:
            chunks.append(cleaned_paragraph)

    return chunks


def load_embedding_model():
    return TextEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
    )


def embed_chunks(embedding_model, chunks):
    return list(
        embedding_model.passage_embed(chunks)
    )


def embed_query(embedding_model, query):
    return list(
        embedding_model.query_embed(query)
    )[0]


def cosine_similarity(first_vector, second_vector):
    numerator = np.dot(
        first_vector,
        second_vector,
    )

    denominator = (
        np.linalg.norm(first_vector)
        * np.linalg.norm(second_vector)
    )

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def search_chunks(
    query,
    chunks,
    chunk_embeddings,
    embedding_model,
    limit=3,
):
    query_embedding = embed_query(
        embedding_model,
        query,
    )

    scored_chunks = []

    for chunk, chunk_embedding in zip(
        chunks,
        chunk_embeddings,
    ):
        score = cosine_similarity(
            query_embedding,
            chunk_embedding,
        )

        scored_chunks.append(
            (score, chunk)
        )

    scored_chunks.sort(
        key=lambda result: result[0],
        reverse=True,
    )

    return scored_chunks[:limit]
