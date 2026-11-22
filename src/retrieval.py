from hashlib import sha256
from pathlib import Path

from fastembed import TextEmbedding


EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


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

    for document in documents:
        paragraphs = document["text"].split(
            "\n\n"
        )

        chunk_index = 0

        for paragraph in paragraphs:
            cleaned_paragraph = (
                paragraph.strip()
            )

            if cleaned_paragraph:
                chunks.append(
                    {
                        "source": document[
                            "source"
                        ],
                        "file_hash": document[
                            "file_hash"
                        ],
                        "chunk_index": chunk_index,
                        "text": cleaned_paragraph,
                    }
                )

                chunk_index += 1

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
