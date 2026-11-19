from retrieval import (
    chunk_documents,
    load_documents,
)


def test_load_documents_loads_multiple_text_files(tmp_path):
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

    documents = load_documents(tmp_path)

    assert len(documents) == 2

    assert documents[0]["source"] == "first.txt"
    assert documents[0]["text"] == "First document."

    assert documents[1]["source"] == "second.txt"
    assert documents[1]["text"] == "Second document."


def test_chunk_documents_preserves_source_metadata():
    documents = [
        {
            "source": "dune.txt",
            "text": (
                "Paul can see possible futures.\n\n"
                "Fremen wear stillsuits."
            ),
        },
        {
            "source": "docker.txt",
            "text": (
                "Bind mounts preserve host data."
            ),
        },
    ]

    chunks = chunk_documents(documents)

    assert len(chunks) == 3

    assert chunks[0] == {
        "source": "dune.txt",
        "text": "Paul can see possible futures.",
    }

    assert chunks[1] == {
        "source": "dune.txt",
        "text": "Fremen wear stillsuits.",
    }

    assert chunks[2] == {
        "source": "docker.txt",
        "text": "Bind mounts preserve host data.",
    }
