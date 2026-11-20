import chromadb


CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "local_ai_knowledge"


def get_collection(
    path=CHROMA_PATH,
    name=COLLECTION_NAME,
):
    client = chromadb.PersistentClient(
        path=path,
    )

    return client.get_or_create_collection(
        name=name,
        embedding_function=None,
        configuration={
            "hnsw": {
                "space": "cosine",
            }
        },
    )


def index_chunks(
    collection,
    chunks,
    chunk_embeddings,
):
    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for index, (
        chunk,
        chunk_embedding,
    ) in enumerate(
        zip(
            chunks,
            chunk_embeddings,
        )
    ):
        ids.append(
            f"chunk-{index}"
        )

        documents.append(
            chunk["text"]
        )

        metadatas.append(
            {
                "source": chunk["source"],
            }
        )

        if hasattr(
            chunk_embedding,
            "tolist",
        ):
            embedding = (
                chunk_embedding.tolist()
            )
        else:
            embedding = list(
                chunk_embedding
            )

        embeddings.append(
            embedding
        )

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )


def search_collection(
    collection,
    query_embedding,
    limit=3,
):
    collection_size = collection.count()

    if collection_size == 0:
        return []

    if hasattr(
        query_embedding,
        "tolist",
    ):
        query_vector = (
            query_embedding.tolist()
        )
    else:
        query_vector = list(
            query_embedding
        )

    results = collection.query(
        query_embeddings=[
            query_vector
        ],
        n_results=min(
            limit,
            collection_size,
        ),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    scored_chunks = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for (
        document,
        metadata,
        distance,
    ) in zip(
        documents,
        metadatas,
        distances,
    ):
        similarity = (
            1.0 - float(distance)
        )

        chunk = {
            "source": metadata["source"],
            "text": document,
        }

        scored_chunks.append(
            (
                similarity,
                chunk,
            )
        )

    return scored_chunks
