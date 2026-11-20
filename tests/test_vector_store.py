from vector_store import (
    get_collection,
    index_chunks,
    search_collection,
)


def test_chroma_persists_chunks_and_metadata(
    tmp_path,
):
    chroma_path = (
        tmp_path / "chroma"
    )

    collection = get_collection(
        path=chroma_path,
        name="test_collection",
    )

    chunks = [
        {
            "source": "dune.txt",
            "text": "Paul can see possible futures.",
        },
        {
            "source": "docker.txt",
            "text": "Bind mounts preserve host data.",
        },
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    index_chunks(
        collection,
        chunks,
        embeddings,
    )

    assert collection.count() == 2

    reopened_collection = get_collection(
        path=chroma_path,
        name="test_collection",
    )

    assert (
        reopened_collection.count()
        == 2
    )

    results = search_collection(
        reopened_collection,
        query_embedding=[
            1.0,
            0.0,
            0.0,
        ],
        limit=1,
    )

    _, chunk = results[0]

    assert (
        chunk["source"]
        == "dune.txt"
    )

    assert (
        chunk["text"]
        == "Paul can see possible futures."
    )
