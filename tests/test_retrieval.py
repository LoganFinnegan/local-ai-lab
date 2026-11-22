from hashlib import sha256

from retrieval import (
    chunk_documents,
    load_documents,
)


def test_load_documents_loads_multiple_text_files(
    tmp_path,
):
    first_file = tmp_path / "first.txt"
    second_file = tmp_path / "second.txt"

    first_file.write_text(
        "First document.",
        encoding="utf-8",
    )

    second_file.write_text(
        "Second document.",
        encoding="utf-8",
    )

    documents = load_documents(
        tmp_path
    )

    assert len(documents) == 2

    assert (
        documents[0]["source"]
        == "first.txt"
    )

    assert (
        documents[0]["text"]
        == "First document."
    )

    assert (
        documents[0]["file_hash"]
        == sha256(
            b"First document."
        ).hexdigest()
    )

    assert (
        documents[1]["source"]
        == "second.txt"
    )

    assert (
        documents[1]["text"]
        == "Second document."
    )


def test_chunk_documents_preserves_source_metadata():
    documents = [
        {
            "source": "dune.txt",
            "file_hash": "dune-hash",
            "text": (
                "Paul can see possible futures.\n\n"
                "Fremen wear stillsuits."
            ),
        },
        {
            "source": "docker.txt",
            "file_hash": "docker-hash",
            "text": (
                "Bind mounts preserve host data."
            ),
        },
    ]

    chunks = chunk_documents(
        documents
    )

    assert len(chunks) == 2

    assert chunks[0] == {
        "source": "dune.txt",
        "file_hash": "dune-hash",
        "chunk_index": 0,
        "text": (
            "Paul can see possible futures. "
            "Fremen wear stillsuits."
        ),
    }

    assert chunks[1] == {
        "source": "docker.txt",
        "file_hash": "docker-hash",
        "chunk_index": 0,
        "text": (
            "Bind mounts preserve host data."
        ),
    }


def test_chunk_documents_handles_crlf_line_endings():
    documents = [
        {
            "source": "book.txt",
            "file_hash": "book-hash",
            "text": (
                "First paragraph.\r\n"
                "\r\n"
                "Second paragraph."
            ),
        }
    ]

    chunks = chunk_documents(
        documents
    )

    assert len(chunks) == 1

    assert chunks[0]["text"] == (
        "First paragraph. "
        "Second paragraph."
    )
