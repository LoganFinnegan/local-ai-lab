import pytest

from vector_store import (
    get_chunk_by_source_and_index,
    get_collection,
    index_chunks,
    search_collection,
    sync_collection,
)


class FakeEmbeddingModel:
    def __init__(self):
        self.passage_calls = 0

    def passage_embed(
        self,
        texts,
    ):
        self.passage_calls += 1

        return [
            [
                float(
                    len(text)
                ),
                1.0,
                0.0,
            ]
            for text in texts
        ]


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
            "file_hash": "dune-hash",
            "chunk_index": 0,
            "text": (
                "Paul can see possible futures."
            ),
        },
        {
            "source": "docker.txt",
            "file_hash": "docker-hash",
            "chunk_index": 0,
            "text": (
                "Bind mounts preserve host data."
            ),
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


def test_sync_collection_skips_unchanged_document(
    tmp_path,
):
    collection = get_collection(
        path=tmp_path / "chroma",
        name="sync_unchanged",
    )

    embedding_model = (
        FakeEmbeddingModel()
    )

    documents = [
        {
            "source": "notes.txt",
            "file_hash": "hash-one",
            "text": "Original text.",
        }
    ]

    first_summary = sync_collection(
        collection,
        documents,
        embedding_model,
    )

    assert first_summary["added"] == [
        "notes.txt"
    ]

    assert (
        embedding_model.passage_calls
        == 1
    )

    second_summary = sync_collection(
        collection,
        documents,
        embedding_model,
    )

    assert (
        second_summary["unchanged"]
        == ["notes.txt"]
    )

    assert (
        embedding_model.passage_calls
        == 1
    )

    assert collection.count() == 1


def test_sync_collection_replaces_changed_document(
    tmp_path,
):
    collection = get_collection(
        path=tmp_path / "chroma",
        name="sync_changed",
    )

    embedding_model = (
        FakeEmbeddingModel()
    )

    original_documents = [
        {
            "source": "notes.txt",
            "file_hash": "old-hash",
            "text": (
                "First chunk.\n\n"
                "Second chunk."
            ),
        }
    ]

    sync_collection(
        collection,
        original_documents,
        embedding_model,
    )

    assert collection.count() == 1

    changed_documents = [
        {
            "source": "notes.txt",
            "file_hash": "new-hash",
            "text": "Replacement text.",
        }
    ]

    summary = sync_collection(
        collection,
        changed_documents,
        embedding_model,
    )

    assert summary["changed"] == [
        "notes.txt"
    ]

    assert collection.count() == 1

    stored = collection.get(
        include=[
            "documents",
            "metadatas",
        ]
    )

    assert stored["documents"] == [
        "Replacement text."
    ]

    assert (
        stored["metadatas"][0][
            "file_hash"
        ]
        == "new-hash"
    )


def test_sync_collection_removes_deleted_document(
    tmp_path,
):
    collection = get_collection(
        path=tmp_path / "chroma",
        name="sync_deleted",
    )

    embedding_model = (
        FakeEmbeddingModel()
    )

    documents = [
        {
            "source": "keep.txt",
            "file_hash": "keep-hash",
            "text": "Keep this.",
        },
        {
            "source": "delete.txt",
            "file_hash": "delete-hash",
            "text": "Delete this.",
        },
    ]

    sync_collection(
        collection,
        documents,
        embedding_model,
    )

    assert collection.count() == 2

    remaining_documents = [
        documents[0]
    ]

    summary = sync_collection(
        collection,
        remaining_documents,
        embedding_model,
    )

    assert summary["deleted"] == [
        "delete.txt"
    ]

    assert collection.count() == 1

    stored = collection.get(
        include=[
            "metadatas",
        ]
    )

    assert (
        stored["metadatas"][0][
            "source"
        ]
        == "keep.txt"
    )


def test_exact_lookup_returns_correct_chunk(
    tmp_path,
):
    collection = get_collection(
        path=tmp_path / "chroma",
        name="exact_lookup",
    )

    chunks = [
        {
            "source": "frankenstein.txt",
            "file_hash": "frankenstein-hash",
            "chunk_index": 136,
            "text": "This is passage 136.",
        },
        {
            "source": "frankenstein.txt",
            "file_hash": "frankenstein-hash",
            "chunk_index": 137,
            "text": "This is passage 137.",
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

    result = get_chunk_by_source_and_index(
        collection,
        "frankenstein.txt",
        137,
    )

    assert result == {
        "source": "frankenstein.txt",
        "chunk_index": 137,
        "text": "This is passage 137.",
    }


def test_exact_lookup_returns_none_when_missing(
    tmp_path,
):
    collection = get_collection(
        path=tmp_path / "chroma",
        name="missing_lookup",
    )

    result = get_chunk_by_source_and_index(
        collection,
        "frankenstein.txt",
        999,
    )

    assert result is None


@pytest.mark.parametrize(
    ("source", "chunk_index"),
    [
        ("", 137),
        (None, 137),
        ("frankenstein.txt", -1),
        ("frankenstein.txt", "137"),
        ("frankenstein.txt", True),
    ],
)
def test_exact_lookup_rejects_invalid_identifiers(
    tmp_path,
    source,
    chunk_index,
):
    collection = get_collection(
        path=tmp_path / "chroma",
        name="invalid_lookup",
    )

    with pytest.raises(ValueError):
        get_chunk_by_source_and_index(
            collection,
            source,
            chunk_index,
        )
