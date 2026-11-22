from retrieval import (
    chunk_documents,
    load_documents,
)


KNOWLEDGE_DIRECTORY = "knowledge"


documents = load_documents(
    KNOWLEDGE_DIRECTORY
)

chunks = chunk_documents(
    documents
)

chunk_sizes = [
    len(
        chunk["text"].split()
    )
    for chunk in chunks
]

print(
    f"Documents: {len(documents)}"
)

print(
    f"Chunks: {len(chunks)}"
)

if not chunk_sizes:
    print("No chunks found.")
    raise SystemExit

print(
    f"Smallest chunk: "
    f"{min(chunk_sizes)} words"
)

print(
    f"Largest chunk: "
    f"{max(chunk_sizes)} words"
)

print(
    f"Average chunk: "
    f"{sum(chunk_sizes) / len(chunk_sizes):.1f} words"
)

tiny_chunks = [
    size
    for size in chunk_sizes
    if size < 20
]

large_chunks = [
    size
    for size in chunk_sizes
    if size > 500
]

print(
    f"Chunks under 20 words: "
    f"{len(tiny_chunks)}"
)

print(
    f"Chunks over 500 words: "
    f"{len(large_chunks)}"
)
