"""Test that the chat application can start without a running LLM server."""

import runpy

from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

import chat_session
import database


@pytest.mark.parametrize(
    "selection, expected_message",
    [
        ("p", "Started a private conversation."),
        ("", "Started a new saved conversation."),
    ],
)
def test_chat_client_starts_and_exits_without_server(
    monkeypatch,
    capsys,
    selection,
    expected_message,
):
    """
    Verify that the actual chat application starts.

    Replace external dependencies so this test does
    not require the LLM server or modify the database.
    """

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Startup must not send a prompt or save messages."
        )

    # Prevent access to the actual conversation database.
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
        chat_session,
        "create_conversation",
        forbidden,
    )

    monkeypatch.setattr(
        chat_session,
        "save_message",
        forbidden,
    )

    # No real model requests should occur.
    monkeypatch.setattr(
        requests,
        "post",
        forbidden,
    )

    # Pretend the inference server reports a valid context window.
    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "default_generation_settings": {
                    "n_ctx": 32768,
                },
            },
        ),
    )

    # First input selects the conversation type.
    # Second input immediately exits the chat.
    choices = iter((selection, "exit"))

    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: next(choices),
    )

    # Execute the real application entry point.
    script_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "llm_client.py"
    )

    runpy.run_path(
        str(script_path),
        run_name="__main__",
    )

    # Confirm that the correct conversation type started.
    output = capsys.readouterr().out

    assert expected_message in output
