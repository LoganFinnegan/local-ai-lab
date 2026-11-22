"""Automated retrieval and RAG answer evaluation."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

from citation_validation import check_citation_identifiers
from rag import citation_for_chunk


DEFAULT_CASES_PATH = Path("evals/frankenstein.json")
DEFAULT_REPORT_DIRECTORY = Path("data/rag_evaluations")
DEFAULT_RETRIEVAL_LIMIT = 10


def matches_groups(text, groups):
    normalized = text.casefold()

    return all(
        any(
            phrase.casefold() in normalized
            for phrase in group
        )
        for group in groups
    )


def find_evidence_rank(results, case):
    groups = case.get("evidence_groups", [])

    if not groups:
        return None

    for rank, (_score, chunk) in enumerate(
        results,
        start=1,
    ):
        if chunk["source"] != case["source"]:
            continue

        if matches_groups(chunk["text"], groups):
            return rank

    return None


def check_answer(answer, case):
    required_groups = case.get("answer_groups", [])

    missing_groups = [
        group
        for group in required_groups
        if not matches_groups(answer, [group])
    ]

    flagged_phrases = [
        phrase
        for phrase in case.get("review_phrases", [])
        if phrase.casefold() in answer.casefold()
    ]

    if missing_groups:
        status = "FAIL"

    elif flagged_phrases:
        status = "REVIEW"

    else:
        status = "PASS"

    return {
        "status": status,
        "missing_groups": missing_groups,
        "flagged_phrases": flagged_phrases,
    }


def check_citation_support(answer, case, retrieved_chunks):
    """
    Check citation IDs and curated textual evidence anchors.

    This is a deterministic regression check, not semantic fact-checking.
    A matching phrase does not prove that a claim is supported.
    """

    identifiers = check_citation_identifiers(
        answer,
        retrieved_chunks,
    )

    valid = identifiers["valid_citations"]
    invalid = identifiers["invalid_citations"]

    cited_identifiers = set(valid)

    cited_chunks = [
        chunk
        for chunk in retrieved_chunks
        if citation_for_chunk(chunk) in cited_identifiers
    ]

    missing_support_groups = [
        group
        for group in case.get("citation_support_groups", [])
        if not any(
            matches_groups(chunk["text"], [group])
            for chunk in cited_chunks
        )
    ]

    citations_required = case.get("citations_required", True)

    if invalid or missing_support_groups:
        status = "FAIL"
    elif citations_required and not valid:
        status = "FAIL"
    elif not valid:
        status = "N/A"
    else:
        status = "PASS"

    return {
        "status": status,
        "valid_citations": valid,
        "invalid_citations": invalid,
        "missing_support_groups": missing_support_groups,
    }


def load_cases(path):
    with path.open(encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list) or not cases:
        raise ValueError(
            "Evaluation file must contain a non-empty list."
        )

    for case in cases:
        for required_key in (
            "id",
            "question",
            "source",
            "answer_groups",
        ):
            if required_key not in case:
                raise ValueError(
                    f"Case is missing {required_key}: {case}"
                )

    return cases


def write_report(report, directory):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    path = directory / f"frankenstein-{timestamp}.json"

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")

    return path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate local RAG retrieval and answers."
    )

    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Test retrieval without calling Qwen.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_RETRIEVAL_LIMIT,
        help="Number of retrieved chunks.",
    )

    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
    )

    args = parser.parse_args()

    if args.limit < 1:
        parser.error(
            "--limit must be at least 1"
        )

    from retrieval import (
        EMBEDDING_MODEL_NAME,
        embed_query,
        load_documents,
        load_embedding_model,
    )

    from vector_store import (
        get_collection,
        search_collection,
        sync_collection,
    )

    cases = load_cases(args.cases)

    documents = load_documents(
        "knowledge"
    )

    print("Loading embedding model...")
    embedding_model = load_embedding_model()

    collection = get_collection()

    sync_summary = sync_collection(
        collection,
        documents,
        embedding_model,
    )

    report = {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "retrieval_limit": args.limit,
        "retrieval_only": args.retrieval_only,
        "collection_count": collection.count(),
        "sync_summary": sync_summary,
        "cases": [],
        "citation_check_note": (
            "Citation checks verify identifiers and selected textual "
            "anchors only; they do not establish factual entailment."
        ),
    }

    print(
        f"Knowledge files: {len(documents)}"
    )

    print(
        f"Indexed chunks: {collection.count()}"
    )

    print(
        f"Cases: {len(cases)} | "
        f"Retrieval limit: {args.limit}"
    )

    print(
        "Answer checks are heuristics, "
        "not proof of factual accuracy.\n"
    )

    if not args.retrieval_only:
        import requests
        import config
        from rag import build_rag_messages

    for index, case in enumerate(
        cases,
        start=1,
    ):
        query_embedding = embed_query(
            embedding_model,
            case["question"],
        )

        results = search_collection(
            collection,
            query_embedding,
            limit=args.limit,
        )

        evidence_rank = find_evidence_rank(
            results,
            case,
        )

        has_evidence_rule = bool(
            case.get("evidence_groups")
        )

        if evidence_rank is not None:
            evidence_status = "PASS"
        elif has_evidence_rule:
            evidence_status = "MISS"
        else:
            evidence_status = "N/A"

        item = {
            "id": case["id"],
            "question": case["question"],
            "evidence_rank": evidence_rank,
            "evidence_status": evidence_status,
            "retrieved": [
                {
                    "rank": rank,
                    "similarity": round(
                        float(score),
                        4,
                    ),
                    "source": chunk["source"],
                    "chunk_index": chunk.get(
                        "chunk_index"
                    ),
                    "text": chunk["text"],
                }
                for rank, (score, chunk) in enumerate(
                    results,
                    start=1,
                )
            ],
            "answer": None,
            "answer_check": {
                "status": "SKIP"
            },
            "citation_check": {
                "status": "SKIP"
            },
        }

        if not args.retrieval_only:
            retrieved_chunks = [
                chunk
                for _score, chunk in results
            ]

            messages = build_rag_messages(
                case["question"],
                retrieved_chunks,
            )

            payload = {
                "model": config.MODEL_PATH,
                "messages": messages,
                "stream": False,
            }

            start = perf_counter()

            try:
                response = requests.post(
                    config.CHAT_COMPLETIONS_URL,
                    json=payload,
                    timeout=120,
                )

                response.raise_for_status()

                answer = response.json()[
                    "choices"
                ][0][
                    "message"
                ][
                    "content"
                ]

                item["answer"] = answer

                item["answer_check"] = check_answer(
                    answer,
                    case,
                )

                item["citation_check"] = check_citation_support(
                    answer,
                    case,
                    retrieved_chunks,
                )

            except (
                requests.RequestException,
                KeyError,
                IndexError,
                ValueError,
            ) as error:
                item["answer_check"] = {
                    "status": "ERROR",
                    "message": str(error),
                }

            item["inference_seconds"] = round(
                perf_counter() - start,
                2,
            )

        report["cases"].append(item)

        if evidence_rank is not None:
            retrieval_label = (
                f"{evidence_status} "
                f"(rank {evidence_rank})"
            )
        else:
            retrieval_label = evidence_status

        print(
            f"{index}. {case['id']}: "
            f"evidence {retrieval_label}, "
            f"answer {item['answer_check']['status']}, "
            f"citations {item['citation_check']['status']}"
        )

        if item["answer_check"].get(
            "flagged_phrases"
        ):
            print(
                f"   Review: "
                f"{item['answer_check']['flagged_phrases']}"
            )

        if item["citation_check"].get("missing_support_groups"):
            print(
                "   Cited evidence missing: "
                f"{item['citation_check']['missing_support_groups']}"
            )

    path = write_report(
        report,
        DEFAULT_REPORT_DIRECTORY,
    )

    print(
        f"\nSaved report: {path}"
    )

    print(
        "Review saved answers when accuracy matters; "
        "keyword checks can miss errors."
    )


if __name__ == "__main__":
    main()
