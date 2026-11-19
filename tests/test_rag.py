from rag import build_rag_messages


def test_build_rag_messages_includes_query_and_retrieved_chunks():
    query = (
        "Who can see things before they happen?"
    )

    retrieved_chunks = [
        {
            "source": "dune.txt",
            "text": (
                "Paul Atreides can perceive possible "
                "future events after exposure to spice."
            ),
        },
        {
            "source": "dune.txt",
            "text": (
                "Guild Navigators use spice to safely "
                "guide interstellar travel."
            ),
        },
    ]

    messages = build_rag_messages(
        query,
        retrieved_chunks,
    )

    assert len(messages) == 2

    assert messages[0]["role"] == "system"

    assert messages[1]["role"] == "user"

    assert query in messages[1]["content"]

    for chunk in retrieved_chunks:
        assert (
            chunk["source"]
            in messages[1]["content"]
        )

        assert (
            chunk["text"]
            in messages[1]["content"]
        )
