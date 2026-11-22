import requests

import config

from rag import build_rag_messages

from retrieval import (
    embed_query,
    load_documents,
    load_embedding_model,
)

from vector_store import (
    get_collection,
    search_collection,
    sync_collection,
)


KNOWLEDGE_DIRECTORY = "knowledge"
RETRIEVAL_LIMIT = 3


documents = load_documents(
    KNOWLEDGE_DIRECTORY
)

print(
    f"Loaded {len(documents)} documents."
)

print("Loading embedding model...")
embedding_model = load_embedding_model()

collection = get_collection()

print("Syncing knowledge files...")

sync_summary = sync_collection(
    collection,
    documents,
    embedding_model,
)

print(
    f"Added: {len(sync_summary['added'])}"
)

print(
    f"Changed: {len(sync_summary['changed'])}"
)

print(
    f"Deleted: {len(sync_summary['deleted'])}"
)

print(
    f"Unchanged: {len(sync_summary['unchanged'])}"
)

print(
    f"Chroma contains "
    f"{collection.count()} chunks."
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
            f"Chunk: "
            f"{chunk.get('chunk_index', 'unknown')}"
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
