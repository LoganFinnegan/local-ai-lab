def build_rag_messages(
    query,
    retrieved_chunks,
):
    reference_sections = []

    for chunk in retrieved_chunks:
        reference_sections.append(
            (
                f"Source: {chunk['source']}\n"
                f"{chunk['text']}"
            )
        )

    context = "\n\n".join(
        reference_sections
    )

    return [
        {
            "role": "system",
            "content": (
                "Answer the user's question using only the supplied "
                "reference material. If the reference material does not "
                "contain enough information to answer, say that you "
                "cannot answer from the provided material."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Reference material:\n\n{context}\n\n"
                f"Question:\n{query}"
            ),
        },
    ]
