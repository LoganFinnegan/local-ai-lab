"""Ensure quit commands and empty input never reach the inference server."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

import chat_session
import database


@pytest.mark.parametrize(
    "selection",
    [
        "",
        "p",
        "21",
    ],
)
@pytest.mark.parametrize(
    "quit_command",
    [
        "q",
        "Q",
    ],
)
def test_q_exits_chat_and_blank_input_is_ignored(
    monkeypatch,
    selection,
    quit_command,
):
    """
    Blank input and Q must not call the model,
    create conversations, or save messages.

    Test new saved, private, and resumed conversations.
    """

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Blank input and Q must not "
            "call the model or save messages."
        )

    loaded_ids = []

    def load_active_messages(conversation_id):
        loaded_ids.append(conversation_id)

        assert conversation_id == 21

        return [
            (
                "user",
                "Existing question",
            ),
            (
                "assistant",
                "Existing answer",
            ),
        ]

    monkeypatch.setattr(
        database,
        "initialize_database",
        lambda: None,
    )

    monkeypatch.setattr(
        database,
        "list_conversations",
        lambda **kwargs: [],
    )

    monkeypatch.setattr(
        database,
        "conversation_exists",
        lambda number: number == 21,
    )

    monkeypatch.setattr(
        database,
        "load_messages",
        load_active_messages,
    )

    monkeypatch.setattr(
        chat_session,
        "create_conversation",
        forbidden,
    )

    monkeypatch.setattr(
        chat_session,
        "save_message",
        forbidden,
    )

    monkeypatch.setattr(
        requests,
        "post",
        forbidden,
    )

    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "default_generation_settings": {
                    "n_ctx": 32768,
                }
            },
        ),
    )

    commands = iter(
        (
            selection,
            "   ",
            quit_command,
        )
    )

    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: next(commands),
    )

    script_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "llm_client.py"
    )

    runpy.run_path(
        str(script_path),
        run_name="__main__",
    )

    expected_ids = (
        [21]
        if selection == "21"
        else []
    )

    assert loaded_ids == expected_ids
