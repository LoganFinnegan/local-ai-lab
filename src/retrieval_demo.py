from retrieval import (
    chunk_text,
    embed_chunks,
    load_embedding_model,
    load_text,
    search_chunks,
)


DOCUMENT_PATH = "knowledge/sample.txt"


text = load_text(DOCUMENT_PATH)
chunks = chunk_text(text)

print(f"Loaded {len(chunks)} chunks.")

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
        "\nAsk about the document: "
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
        print(f"\nSimilarity: {score:.3f}")
        print(chunk)
