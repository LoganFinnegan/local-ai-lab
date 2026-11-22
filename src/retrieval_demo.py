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
    )

    print("\nRetrieved chunks:")

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
