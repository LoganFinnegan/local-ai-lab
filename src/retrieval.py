from hashlib import sha256
from pathlib import Path

from fastembed import TextEmbedding


EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE_WORDS = 250
CHUNK_OVERLAP_WORDS = 50


def load_documents(directory_path):
    directory = Path(directory_path)

    documents = []

    for file_path in sorted(
        directory.glob("*.txt")
    ):
        file_bytes = file_path.read_bytes()

        documents.append(
            {
                "source": file_path.name,
                "text": file_bytes.decode(
                    "utf-8"
                ),
                "file_hash": sha256(
                    file_bytes
                ).hexdigest(),
            }
        )

    return documents


def chunk_documents(documents):
    chunks = []

    step_size = (
        CHUNK_SIZE_WORDS
        - CHUNK_OVERLAP_WORDS
    )

    for document in documents:
        normalized_text = (
            document["text"]
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )

        words = normalized_text.split()

        chunk_index = 0

        for start in range(
            0,
            len(words),
            step_size,
        ):
            end = (
                start
                + CHUNK_SIZE_WORDS
            )

            chunk_words = words[
                start:end
            ]

            if not chunk_words:
                continue

            chunk_text = " ".join(
                chunk_words
            )

            chunks.append(
                {
                    "source": document[
                        "source"
                    ],
                    "file_hash": document[
                        "file_hash"
                    ],
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                }
            )

            chunk_index += 1

            if end >= len(words):
                break

    return chunks


def load_embedding_model():
    return TextEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        cache_dir="data/fastembed_cache",
    )


def embed_chunks(
    embedding_model,
    chunks,
):
    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    return list(
        embedding_model.passage_embed(
            chunk_texts
        )
    )


def embed_query(
    embedding_model,
    query,
):
    return list(
        embedding_model.query_embed(
            query
        )
    )[0]
