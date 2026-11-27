"""Test conversation history formatting and privacy."""

import pytest

from chat_history import HISTORY_USAGE, history_response


CURRENT_MESSAGES = [
    {
        "role": "system",
        "content": "Secret internal instructions",
    },
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
    {
        "role": "assistant",
        "content": "Second answer",
    },
]


def test_history_shows_current_conversation_and_excludes_system():
    output = history_response(
        "history",
        CURRENT_MESSAGES,
    )

    assert "First question" in output
    assert "First answer" in output
    assert "Second question" in output
    assert "Second answer" in output

    assert "Secret internal instructions" not in output

    assert (
        output.index("First question")
        < output.index("Second answer")
    )


def test_history_number_shows_last_n_messages():
    output = history_response(
        "history 2",
        CURRENT_MESSAGES,
    )

    assert "First question" not in output
    assert "First answer" not in output

    assert "You:\nSecond question" in output
    assert "Assistant:\nSecond answer" in output

    assert (
        output.index("Second question")
        < output.index("Second answer")
    )


@pytest.mark.parametrize(
    "command",
    [
        "history 0",
        "history -3",
        "history 1.5",
        "history banana",
        "history 2 extra",
        "history ١",
    ],
)
def test_invalid_history_argument(command):
    result = history_response(
        command,
        CURRENT_MESSAGES,
    )

    assert result == HISTORY_USAGE


def test_history_is_case_insensitive():
    result = history_response(
        "HISTORY 1",
        CURRENT_MESSAGES,
    )

    assert "Second answer" in result
    assert "First question" not in result


def test_history_does_not_match_unrelated_text():
    result = history_response(
        "historybook recommendations",
        CURRENT_MESSAGES,
    )

    assert result is None


def test_private_history_does_not_display_messages():
    result = history_response(
        "history",
        CURRENT_MESSAGES,
        private=True,
    )

    assert "scrollback" in result

    assert "First question" not in result
    assert "First answer" not in result
    assert "Secret internal instructions" not in result


def test_empty_conversation_has_no_history():
    result = history_response(
        "history",
        [CURRENT_MESSAGES[0]],
    )

    assert result == (
        "No messages in this conversation yet."
    )


def test_history_does_not_modify_original_messages():
    original_messages = [
        message.copy()
        for message in CURRENT_MESSAGES
    ]

    history_response(
        "history 2",
        CURRENT_MESSAGES,
    )

    assert CURRENT_MESSAGES == original_messages
