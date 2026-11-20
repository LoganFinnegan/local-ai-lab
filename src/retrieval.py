from pathlib import Path

from fastembed import TextEmbedding


EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_documents(directory_path):
    directory = Path(directory_path)

    documents = []

    for file_path in sorted(
        directory.glob("*.txt")
    ):
        documents.append(
            {
                "source": file_path.name,
                "text": file_path.read_text(
                    encoding="utf-8",
                ),
            }
        )

    return documents


def chunk_documents(documents):
    chunks = []

    for document in documents:
        paragraphs = document["text"].split(
            "\n\n"
        )

        for paragraph in paragraphs:
            cleaned_paragraph = (
                paragraph.strip()
            )

            if cleaned_paragraph:
                chunks.append(
                    {
                        "source": document["source"],
                        "text": cleaned_paragraph,
                    }
                )

    return chunks


def load_embedding_model():
    return TextEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
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
