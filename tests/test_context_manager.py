import pytest

from context_manager import fit_messages_to_budget


SYSTEM_MESSAGE = {
    "role": "system",
    "content": "System",
}


def count_messages(messages):
    return len(messages)


def test_returns_full_history_when_it_fits():
    messages = [
        SYSTEM_MESSAGE,
        {
            "role": "user",
            "content": "First question",
        },
        {
            "role": "assistant",
            "content": "First answer",
        },
        {
            "role": "user",
            "content": "Second question",
        },
    ]

    (
        request_messages,
        full_tokens,
        fitted_tokens,
        omitted_messages,
    ) = fit_messages_to_budget(
        messages,
        input_token_budget=4,
        count_tokens=count_messages,
    )

    assert request_messages == messages
    assert full_tokens == 4
    assert fitted_tokens == 4
    assert omitted_messages == 0


def test_omits_oldest_turns_when_history_is_too_large():
    messages = [
        SYSTEM_MESSAGE,
        {
            "role": "user",
            "content": "Old question",
        },
        {
            "role": "assistant",
            "content": "Old answer",
        },
        {
            "role": "user",
            "content": "Recent question",
        },
        {
            "role": "assistant",
            "content": "Recent answer",
        },
        {
            "role": "user",
            "content": "Current question",
        },
    ]

    (
        request_messages,
        full_tokens,
        fitted_tokens,
        omitted_messages,
    ) = fit_messages_to_budget(
        messages,
        input_token_budget=4,
        count_tokens=count_messages,
    )

    assert request_messages == [
        SYSTEM_MESSAGE,
        {
            "role": "user",
            "content": "Recent question",
        },
        {
            "role": "assistant",
            "content": "Recent answer",
        },
        {
            "role": "user",
            "content": "Current question",
        },
    ]

    assert full_tokens == 6
    assert fitted_tokens == 4
    assert omitted_messages == 2


def test_does_not_modify_original_history():
    messages = [
        SYSTEM_MESSAGE,
        {
            "role": "user",
            "content": "Old question",
        },
        {
            "role": "assistant",
            "content": "Old answer",
        },
        {
            "role": "user",
            "content": "Current question",
        },
    ]

    original_messages = [
        dict(message)
        for message in messages
    ]

    fit_messages_to_budget(
        messages,
        input_token_budget=2,
        count_tokens=count_messages,
    )

    assert messages == original_messages


def test_raises_when_system_and_current_prompt_cannot_fit():
    messages = [
        SYSTEM_MESSAGE,
        {
            "role": "user",
            "content": "Current question",
        },
    ]

    def oversized_counter(messages):
        return 100

    with pytest.raises(
        ValueError,
        match="system message and current prompt",
    ):
        fit_messages_to_budget(
            messages,
            input_token_budget=50,
            count_tokens=oversized_counter,
        )


def test_large_history_uses_few_token_count_checks():
    messages = [SYSTEM_MESSAGE]

    for number in range(10_000):
        messages.append({
            "role": "user",
            "content": f"Question {number}",
        })

        messages.append({
            "role": "assistant",
            "content": f"Answer {number}",
        })

    messages.append({
        "role": "user",
        "content": "Current question",
    })

    count_calls = 0

    def tracked_counter(candidate_messages):
        nonlocal count_calls
        count_calls += 1
        return len(candidate_messages)

    (
        request_messages,
        full_tokens,
        fitted_tokens,
        omitted_messages,
    ) = fit_messages_to_budget(
        messages,
        input_token_budget=101,
        count_tokens=tracked_counter,
    )

    assert fitted_tokens <= 101
    assert request_messages[0]["role"] == "system"
    assert request_messages[-1]["content"] == "Current question"
    assert omitted_messages > 0

    assert count_calls <= 20
