from pathlib import Path

from evaluate_rag import (
    check_answer,
    check_citation_support,
    find_evidence_rank,
    load_cases,
    matches_groups,
)


def test_matches_groups_accepts_alternative_phrases():
    text = "Paradise Lost and Plutarch's Lives"

    groups = [
        ["paradise lost"],
        ["plutarch", "other"],
    ]

    assert matches_groups(text, groups)


def test_matches_groups_requires_every_group():
    text = "Paradise Lost"

    groups = [
        ["paradise lost"],
        ["plutarch"],
    ]

    assert not matches_groups(text, groups)


def test_find_evidence_rank_checks_source_and_content():
    case = {
        "source": "frankenstein.txt",
        "evidence_groups": [
            ["Justine"]
        ],
    }

    results = [
        (
            0.9,
            {
                "source": "other.txt",
                "text": "Justine",
            },
        ),
        (
            0.8,
            {
                "source": "frankenstein.txt",
                "text": "The creature fled.",
            },
        ),
        (
            0.7,
            {
                "source": "frankenstein.txt",
                "text": "Justine was tried.",
            },
        ),
    ]

    assert find_evidence_rank(
        results,
        case,
    ) == 3


def test_find_evidence_rank_returns_none_when_missing():
    case = {
        "source": "frankenstein.txt",
        "evidence_groups": [
            ["Plutarch"]
        ],
    }

    results = []

    assert find_evidence_rank(
        results,
        case,
    ) is None


def test_answer_check_flags_misleading_justine_claim():
    case = {
        "answer_groups": [
            ["Justine"]
        ],
        "review_phrases": [
            "ultimately declared innocent"
        ],
    }

    answer = (
        "Justine Moritz was ultimately "
        "declared innocent by Elizabeth."
    )

    result = check_answer(
        answer,
        case,
    )

    assert result["status"] == "REVIEW"


def test_answer_check_fails_missing_required_fact():
    case = {
        "answer_groups": [
            ["convicted", "found guilty"],
            ["executed", "hanged"],
        ],
    }

    answer = "Elizabeth defended Justine."

    result = check_answer(
        answer,
        case,
    )

    assert result["status"] == "FAIL"


def test_unanswerable_answer_with_does_not_mention_passes():
    case = {
        "answer_groups": [
            [
                "not mentioned",
                "does not mention",
            ]
        ],
    }

    answer = (
        "The reference material does not "
        "mention the creature's Social "
        "Security number."
    )

    result = check_answer(
        answer,
        case,
    )

    assert result["status"] == "PASS"


def make_chunk(text, index, source="frankenstein.txt"):
    return {
        "source": source,
        "chunk_index": index,
        "text": text,
    }


def test_cited_chunks_can_jointly_cover_required_evidence():
    chunks = [
        make_chunk("Justine was found guilty.", 140),
        make_chunk("Justine died on the scaffold.", 146),
    ]

    case = {
        "citation_support_groups": [
            ["guilty", "condemned"],
            ["scaffold", "executed"],
        ],
    }

    answer = (
        "Justine was convicted [frankenstein.txt#140] "
        "and executed [frankenstein.txt#146]."
    )

    result = check_citation_support(answer, case, chunks)

    assert result["status"] == "PASS"
    assert result["missing_support_groups"] == []


def test_valid_id_with_unrelated_cited_text_fails():
    chunks = [
        make_chunk("Elizabeth defended Justine.", 137),
        make_chunk("Justine died on the scaffold.", 146),
    ]

    case = {
        "citation_support_groups": [
            ["scaffold", "executed"],
        ],
    }

    answer = "Justine was executed [frankenstein.txt#137]."
    result = check_citation_support(answer, case, chunks)

    assert result["status"] == "FAIL"
    assert result["valid_citations"] == ["[frankenstein.txt#137]"]
    assert result["missing_support_groups"] == [
        ["scaffold", "executed"]
    ]


def test_fabricated_citation_fails():
    chunks = [make_chunk("Justine was condemned.", 140)]

    result = check_citation_support(
        "Justine was condemned [frankenstein.txt#999].",
        {},
        chunks,
    )

    assert result["status"] == "FAIL"
    assert result["invalid_citations"] == [
        "[frankenstein.txt#999]"
    ]


def test_fact_answer_without_citation_fails():
    chunks = [make_chunk("Justine was condemned.", 140)]

    result = check_citation_support(
        "Justine was condemned.",
        {},
        chunks,
    )

    assert result["status"] == "FAIL"


def test_unanswerable_without_citations_is_not_penalized():
    chunks = [make_chunk("The creature departed.", 100)]

    result = check_citation_support(
        "The reference material does not provide that information.",
        {"citations_required": False},
        chunks,
    )

    assert result["status"] == "N/A"


def test_parenthesized_citation_is_accepted_and_normalized():
    chunks = [
        make_chunk("A female who will accompany you.", 252),
    ]

    case = {
        "citation_support_groups": [
            ["female"],
            ["accompany"],
        ],
    }

    result = check_citation_support(
        "The creature requested a companion (frankenstein.txt#252).",
        case,
        chunks,
    )

    assert result["status"] == "PASS"
    assert result["valid_citations"] == ["[frankenstein.txt#252]"]


def test_parenthesized_unretrieved_citation_fails():
    chunks = [make_chunk("Justine defended herself.", 138)]

    result = check_citation_support(
        "Justine defended herself (frankenstein.txt#999).",
        {},
        chunks,
    )

    assert result["status"] == "FAIL"
    assert result["invalid_citations"] == ["[frankenstein.txt#999]"]


def test_mixed_citation_wrappers_are_deduplicated():
    chunks = [make_chunk("Justine was condemned.", 140)]

    result = check_citation_support(
        "[frankenstein.txt#140] (frankenstein.txt#140)",
        {},
        chunks,
    )

    assert result["status"] == "PASS"
    assert result["valid_citations"] == ["[frankenstein.txt#140]"]


def get_frankenstein_case(case_id):
    cases_path = (
        Path(__file__).resolve().parents[1]
        / "evals"
        / "frankenstein.json"
    )

    return next(
        case
        for case in load_cases(cases_path)
        if case["id"] == case_id
    )


def test_elizabeth_answer_accepts_innocence_wording():
    case = get_frankenstein_case("elizabeth_defends_justine")

    result = check_answer(
        "Elizabeth defended Justine and believed in her innocence.",
        case,
    )

    assert result["status"] == "PASS"


def test_william_answer_with_false_exoneration_gets_review():
    case = get_frankenstein_case("william_accusation")

    result = check_answer(
        "Justine was accused of William's murder, "
        "but was ultimately found innocent.",
        case,
    )

    assert result["status"] == "REVIEW"
    assert "ultimately found innocent" in result["flagged_phrases"]


def test_william_answer_with_wrong_evidence_link_gets_review():
    case = get_frankenstein_case("william_accusation")

    result = check_answer(
        "Justine was accused of the murder. A mark found "
        "on William's neck was misinterpreted to implicate Justine.",
        case,
    )

    assert result["status"] == "REVIEW"
