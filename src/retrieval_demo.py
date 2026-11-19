from retrieval import (
    chunk_documents,
    embed_chunks,
    load_documents,
    load_embedding_model,
    search_chunks,
)


KNOWLEDGE_DIRECTORY = "knowledge"


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

print("Embedding document chunks...")
chunk_embeddings = embed_chunks(
    embedding_model,
    chunks,
)

print("Ready.")


while True:
    query = input(
        "\nAsk about the documents: "
    ).strip()

    if query.lower() in {"exit", "quit"}:
        break

    results = search_chunks(
        query,
        chunks,
        chunk_embeddings,
        embedding_model,
    )

    print("\nRetrieved chunks:")

    for score, chunk in results:
        print(
            f"\nSimilarity: {score:.3f}"
        )
        print(
            f"Source: {chunk['source']}"
        )
        print(chunk["text"])
