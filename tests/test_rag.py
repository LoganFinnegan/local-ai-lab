import pytest

from rag import (
    build_rag_messages,
    citation_for_chunk,
    parse_source_command,
)


def test_build_rag_messages_includes_query_and_cited_chunks():
    query = "Who can see things before they happen?"

    retrieved_chunks = [
        {
            "source": "dune.txt",
            "chunk_index": 12,
            "text": (
                "Paul Atreides can perceive possible "
                "future events after exposure to spice."
            ),
        },
        {
            "source": "dune.txt",
            "chunk_index": 13,
            "text": (
                "Guild Navigators use spice to safely "
                "guide interstellar travel."
            ),
        },
    ]

    messages = build_rag_messages(query, retrieved_chunks)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    assert query in messages[1]["content"]

    assert "[dune.txt#12]" in messages[1]["content"]
    assert "[dune.txt#13]" in messages[1]["content"]

    for chunk in retrieved_chunks:
        assert chunk["text"] in messages[1]["content"]


def test_citation_for_chunk_uses_source_and_index():
    chunk = {
        "source": "frankenstein.txt",
        "chunk_index": 137,
    }

    assert citation_for_chunk(chunk) == "[frankenstein.txt#137]"


def test_citation_for_chunk_rejects_missing_source():
    chunk = {
        "chunk_index": 1,
        "text": "Example passage.",
    }

    with pytest.raises(ValueError, match="source"):
        citation_for_chunk(chunk)


def test_citation_for_chunk_rejects_missing_index():
    chunk = {
        "source": "frankenstein.txt",
        "text": "Example passage.",
    }

    with pytest.raises(ValueError, match="chunk_index"):
        citation_for_chunk(chunk)


def test_citation_for_chunk_rejects_negative_index():
    chunk = {
        "source": "frankenstein.txt",
        "chunk_index": -1,
    }

    with pytest.raises(ValueError, match="chunk_index"):
        citation_for_chunk(chunk)


def test_prompt_requires_grounded_citations_and_uncertainty():
    messages = build_rag_messages(
        "What happened?",
        [],
    )

    instructions = messages[0]["content"].lower()

    assert "citations" in instructions
    assert "never invent" in instructions
    assert "cannot answer" in instructions


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        (
            "/source frankenstein.txt#137",
            ("frankenstein.txt", 137),
        ),
        (
            "/source [frankenstein.txt#137]",
            ("frankenstein.txt", 137),
        ),
        (
            "/SOURCE frankenstein.txt#137",
            ("frankenstein.txt", 137),
        ),
    ],
)
def test_parse_source_command_accepts_valid_references(
    prompt,
    expected,
):
    assert parse_source_command(prompt) == expected


def test_parse_source_command_ignores_normal_messages():
    assert parse_source_command(
        "What happened to Justine?"
    ) is None


@pytest.mark.parametrize(
    "prompt",
    [
        "/source",
        "/source frankenstein.txt",
        "/source frankenstein.txt#abc",
        "/source frankenstein.txt#-1",
        "/source [frankenstein.txt#137",
    ],
)
def test_parse_source_command_rejects_invalid_references(
    prompt,
):
    with pytest.raises(ValueError):
        parse_source_command(prompt)
