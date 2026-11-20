import requests

import config

from rag import build_rag_messages

from retrieval import (
    chunk_documents,
    embed_chunks,
    embed_query,
    load_documents,
    load_embedding_model,
)

from vector_store import (
    get_collection,
    index_chunks,
    search_collection,
)


KNOWLEDGE_DIRECTORY = "knowledge"
RETRIEVAL_LIMIT = 3


documents = load_documents(
    KNOWLEDGE_DIRECTORY
)

chunks = chunk_documents(
    documents
)

print(
    f"Loaded {len(documents)} documents "
    f"and {len(chunks)} chunks."
)

print("Loading embedding model...")
embedding_model = load_embedding_model()

collection = get_collection()


if collection.count() == 0:
    print(
        "No saved vector index found."
    )

    print(
        "Embedding document chunks..."
    )

    chunk_embeddings = embed_chunks(
        embedding_model,
        chunks,
    )

    index_chunks(
        collection,
        chunks,
        chunk_embeddings,
    )

    print(
        f"Stored {len(chunks)} chunks "
        "in Chroma."
    )

else:
    print(
        "Loaded existing Chroma collection "
        f"with {collection.count()} chunks."
    )


print("Ready.")


while True:
    query = input(
        "\nAsk about the documents: "
    ).strip()

    if query.lower() in {
        "exit",
        "quit",
    }:
        break

    query_embedding = embed_query(
        embedding_model,
        query,
    )

    results = search_collection(
        collection,
        query_embedding,
        limit=RETRIEVAL_LIMIT,
    )

    print("\nRetrieved chunks:")

    retrieved_chunks = []

    for score, chunk in results:
        print(
            f"\nSimilarity: {score:.3f}"
        )

        print(
            f"Source: {chunk['source']}"
        )

        print(
            chunk["text"]
        )

        retrieved_chunks.append(
            chunk
        )

    messages = build_rag_messages(
        query,
        retrieved_chunks,
    )

    payload = {
        "model": config.MODEL_PATH,
        "messages": messages,
        "stream": False,
    }

    try:
        response = requests.post(
            config.CHAT_COMPLETIONS_URL,
            json=payload,
            timeout=120,
        )

        response.raise_for_status()

    except requests.exceptions.ConnectionError:
        print(
            "\nCould not connect to llama-server."
        )
        continue

    except requests.exceptions.Timeout:
        print(
            "\nThe request timed out."
        )
        continue

    except requests.exceptions.HTTPError as error:
        print(
            f"\nHTTP error: {error}"
        )
        continue

    data = response.json()

    answer = data[
        "choices"
    ][0][
        "message"
    ][
        "content"
    ]

    print("\nRAG answer:")
    print(answer)
