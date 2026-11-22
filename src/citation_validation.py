"""Validate citation identifiers against retrieved document passages."""

import re

from rag import citation_for_chunk


# The prompt requests [filename#N], but Qwen sometimes uses parentheses
# or a comma-separated group: (filename#137, filename#138).
# Capture malformed indexes too, so they count as invalid rather than
# disappearing and being mistaken for a missing citation.
_IDENTIFIER = r"[A-Za-z0-9_./-]+#[^\s,\]\)]*"
_GROUP = rf"{_IDENTIFIER}(?:\s*,\s*{_IDENTIFIER})*"
_CITATION_PATTERN = re.compile(
    rf"\[({_GROUP})\]|\(({_GROUP})\)"
)


def check_citation_identifiers(answer, retrieved_chunks):
    """Check cited IDs, not whether the cited text supports the claims.

    PASS: at least one recognized citation, all from retrieved chunks.
    FAIL: one or more recognized citations are not from retrieved chunks.
    REVIEW: no recognized citations were present.

    Returned IDs always use [filename#N], even if the answer used (...).
    """
    if not isinstance(answer, str):
        raise TypeError("answer must be a string")

    available = {
        citation_for_chunk(chunk)
        for chunk in retrieved_chunks
    }

    valid = []
    invalid = []
    seen = set()

    for match in _CITATION_PATTERN.finditer(answer):
        group = match.group(1) or match.group(2)

        for identifier in group.split(","):
            citation = f"[{identifier.strip()}]"

            if citation in seen:
                continue

            seen.add(citation)

            if citation in available:
                valid.append(citation)
            else:
                invalid.append(citation)

    if invalid:
        status = "FAIL"
    elif valid:
        status = "PASS"
    else:
        status = "REVIEW"

    return {
        "status": status,
        "valid_citations": valid,
        "invalid_citations": invalid,
    }
