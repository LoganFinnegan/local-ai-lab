
from evaluate_rag import (
    check_answer,
    find_evidence_rank,
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
