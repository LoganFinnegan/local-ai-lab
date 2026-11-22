import chromadb

from retrieval import (
    chunk_documents,
    embed_chunks,
)


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
    if not chunks:
        return

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for chunk, chunk_embedding in zip(
        chunks,
        chunk_embeddings,
    ):
        ids.append(
            (
                f"{chunk['source']}:"
                f"{chunk['chunk_index']}"
            )
        )

        documents.append(
            chunk["text"]
        )

        metadatas.append(
            {
                "source": chunk["source"],
                "file_hash": chunk[
                    "file_hash"
                ],
                "chunk_index": chunk[
                    "chunk_index"
                ],
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


def get_indexed_file_hashes(
    collection,
):
    results = collection.get(
        include=[
            "metadatas",
        ],
    )

    hashes_by_source = {}

    for metadata in (
        results["metadatas"] or []
    ):
        if not metadata:
            continue

        source = metadata.get(
            "source"
        )

        if not source:
            continue

        file_hash = metadata.get(
            "file_hash"
        )

        hashes_by_source.setdefault(
            source,
            set(),
        ).add(
            file_hash
        )

    indexed_hashes = {}

    for source, hashes in (
        hashes_by_source.items()
    ):
        if len(hashes) == 1:
            indexed_hashes[source] = next(
                iter(hashes)
            )
        else:
            indexed_hashes[source] = None

    return indexed_hashes


def delete_source(
    collection,
    source,
):
    collection.delete(
        where={
            "source": source,
        }
    )


def sync_collection(
    collection,
    documents,
    embedding_model,
):
    indexed_hashes = (
        get_indexed_file_hashes(
            collection
        )
    )

    current_documents = {
        document["source"]: document
        for document in documents
    }

    summary = {
        "added": [],
        "changed": [],
        "deleted": [],
        "unchanged": [],
    }

    indexed_sources = set(
        indexed_hashes
    )

    current_sources = set(
        current_documents
    )

    deleted_sources = (
        indexed_sources
        - current_sources
    )

    for source in sorted(
        deleted_sources
    ):
        delete_source(
            collection,
            source,
        )

        summary["deleted"].append(
            source
        )

    for source, document in (
        current_documents.items()
    ):
        current_hash = document[
            "file_hash"
        ]

        if (
            source in indexed_hashes
            and indexed_hashes[source]
            == current_hash
        ):
            summary[
                "unchanged"
            ].append(
                source
            )

            continue

        if source in indexed_hashes:
            delete_source(
                collection,
                source,
            )

            summary["changed"].append(
                source
            )

        else:
            summary["added"].append(
                source
            )

        chunks = chunk_documents(
            [
                document
            ]
        )

        if not chunks:
            continue

        chunk_embeddings = embed_chunks(
            embedding_model,
            chunks,
        )

        index_chunks(
            collection,
            chunks,
            chunk_embeddings,
        )

    return summary


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

        if "chunk_index" in metadata:
            chunk["chunk_index"] = (
                metadata["chunk_index"]
            )

        scored_chunks.append(
            (
                similarity,
                chunk,
            )
        )

    return scored_chunks


def get_chunk_by_source_and_index(
    collection,
    source,
    chunk_index,
):
    """
    Retrieve one exact passage from Chroma
    using its source and chunk index.

    Return None when the passage is not found.
    """

    if not isinstance(source, str) or not source:
        raise ValueError(
            "A valid source filename is required."
        )

    if (
        isinstance(chunk_index, bool)
        or not isinstance(chunk_index, int)
        or chunk_index < 0
    ):
        raise ValueError(
            "chunk_index must be a non-negative integer."
        )

    chunk_id = f"{source}:{chunk_index}"

    results = collection.get(
        ids=[chunk_id],
        include=[
            "documents",
            "metadatas",
        ],
    )

    if not results["ids"]:
        return None

    document = results["documents"][0]
    metadata = results["metadatas"][0]

    if (
        not metadata
        or metadata.get("source") != source
        or metadata.get("chunk_index") != chunk_index
    ):
        raise ValueError(
            "Stored chunk metadata does not match "
            "the requested citation."
        )

    return {
        "source": source,
        "chunk_index": chunk_index,
        "text": document,
    }
