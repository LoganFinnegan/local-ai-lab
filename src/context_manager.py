def fit_messages_to_budget(
    messages,
    input_token_budget,
    count_tokens,
):
    full_input_tokens = count_tokens(messages)

    if full_input_tokens <= input_token_budget:
        return (
            list(messages),
            full_input_tokens,
            full_input_tokens,
            0,
        )

    user_message_indexes = [
        index
        for index in range(1, len(messages))
        if messages[index]["role"] == "user"
    ]

    low = 0
    high = len(user_message_indexes) - 1

    best_start_index = None
    best_input_tokens = None

    while low <= high:
        middle = (low + high) // 2
        start_index = user_message_indexes[middle]

        candidate_messages = [
            messages[0],
            *messages[start_index:],
        ]

        candidate_input_tokens = count_tokens(
            candidate_messages
        )

        if candidate_input_tokens <= input_token_budget:
            best_start_index = start_index
            best_input_tokens = candidate_input_tokens

            high = middle - 1

        else:
            low = middle + 1

    if best_start_index is None:
        raise ValueError(
            "The system message and current prompt are larger "
            "than the available input-token budget."
        )

    request_messages = [
        messages[0],
        *messages[best_start_index:],
    ]

    omitted_messages = best_start_index - 1

    return (
        request_messages,
        full_input_tokens,
        best_input_tokens,
        omitted_messages,
    )
