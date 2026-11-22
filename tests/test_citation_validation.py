import pytest

from citation_validation import check_citation_identifiers


def make_chunk(source, chunk_index):
    return {
        "source": source,
        "chunk_index": chunk_index,
        "text": "Example source passage.",
    }


def test_accepts_valid_citation():
    chunks = [
        make_chunk("frankenstein.txt", 146),
    ]

    answer = (
        "Justine was executed "
        "[frankenstein.txt#146]."
    )

    result = check_citation_identifiers(
        answer,
        chunks,
    )

    assert result["status"] == "PASS"

    assert result["valid_citations"] == [
        "[frankenstein.txt#146]"
    ]

    assert result["invalid_citations"] == []


def test_accepts_multiple_valid_citations():
    chunks = [
        make_chunk("frankenstein.txt", 137),
        make_chunk("frankenstein.txt", 140),
        make_chunk("frankenstein.txt", 146),
    ]

    answer = (
        "Elizabeth defended Justine "
        "[frankenstein.txt#137]. "
        "Justine confessed "
        "[frankenstein.txt#140]. "
        "She was executed "
        "[frankenstein.txt#146]."
    )

    result = check_citation_identifiers(
        answer,
        chunks,
    )

    assert result["status"] == "PASS"

    assert len(result["valid_citations"]) == 3

    assert result["invalid_citations"] == []


def test_rejects_citation_not_in_retrieved_chunks():
    chunks = [
        make_chunk("frankenstein.txt", 146),
    ]

    answer = (
        "Justine was executed "
        "[frankenstein.txt#999]."
    )

    result = check_citation_identifiers(
        answer,
        chunks,
    )

    assert result["status"] == "FAIL"

    assert result["invalid_citations"] == [
        "[frankenstein.txt#999]"
    ]


def test_rejects_unretrieved_source():
    chunks = [
        make_chunk("frankenstein.txt", 146),
    ]

    answer = (
        "The answer appears in "
        "[dune.txt#146]."
    )

    result = check_citation_identifiers(
        answer,
        chunks,
    )

    assert result["status"] == "FAIL"

    assert result["invalid_citations"] == [
        "[dune.txt#146]"
    ]


def test_rejects_malformed_citation():
    chunks = [
        make_chunk("frankenstein.txt", 146),
    ]

    answer = (
        "Justine was executed "
        "[frankenstein.txt#unknown]."
    )

    result = check_citation_identifiers(
        answer,
        chunks,
    )

    assert result["status"] == "FAIL"

    assert result["invalid_citations"] == [
        "[frankenstein.txt#unknown]"
    ]


def test_missing_citations_requires_review():
    chunks = [
        make_chunk("frankenstein.txt", 146),
    ]

    answer = (
        "The reference material does not "
        "contain the requested information."
    )

    result = check_citation_identifiers(
        answer,
        chunks,
    )

    assert result["status"] == "REVIEW"

    assert result["valid_citations"] == []

    assert result["invalid_citations"] == []


def test_duplicate_citations_are_reported_once():
    chunks = [
        make_chunk("frankenstein.txt", 146),
    ]

    answer = (
        "Justine was executed "
        "[frankenstein.txt#146]. "
        "The passage confirms her death "
        "[frankenstein.txt#146]."
    )

    result = check_citation_identifiers(
        answer,
        chunks,
    )

    assert result["status"] == "PASS"

    assert result["valid_citations"] == [
        "[frankenstein.txt#146]"
    ]


def test_mixed_valid_and_invalid_citations_fail():
    chunks = [
        make_chunk("frankenstein.txt", 146),
    ]

    answer = (
        "Justine was executed "
        "[frankenstein.txt#146]. "
        "Additional evidence appears in "
        "[frankenstein.txt#999]."
    )

    result = check_citation_identifiers(
        answer,
        chunks,
    )

    assert result["status"] == "FAIL"

    assert result["valid_citations"] == [
        "[frankenstein.txt#146]"
    ]

    assert result["invalid_citations"] == [
        "[frankenstein.txt#999]"
    ]


def test_rejects_non_string_answer():
    chunks = [
        make_chunk("frankenstein.txt", 146),
    ]

    with pytest.raises(
        TypeError,
        match="answer must be a string",
    ):
        check_citation_identifiers(
            None,
            chunks,
        )
