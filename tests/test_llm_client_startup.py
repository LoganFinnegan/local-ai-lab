"""Test the interactive chat entry point without a running LLM server."""

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
    """Catch broken imports, missing functions, and startup crashes."""

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Startup must not send a prompt or save messages."
        )

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

    choices = iter((selection, "exit"))

    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: next(choices),
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

    assert expected_message in capsys.readouterr().out


@pytest.mark.parametrize(
    "exit_command",
    [
        "q",
        "quit",
        "exit",
    ],
)
def test_startup_exit_does_not_require_llm_server(
    monkeypatch,
    capsys,
    exit_command,
):
    """Leaving the initial menu must not contact the model."""

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Startup exit must not contact the model "
            "or access saved messages."
        )

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
        "load_messages",
        forbidden,
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
        "get",
        forbidden,
    )

    monkeypatch.setattr(
        requests,
        "post",
        forbidden,
    )

    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: exit_command,
    )

    script_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "llm_client.py"
    )

    with pytest.raises(SystemExit) as result:
        runpy.run_path(
            str(script_path),
            run_name="__main__",
        )

    assert result.value.code == 0

    output = capsys.readouterr().out

    assert "Enter Q to exit" in output


def test_history_uses_only_active_saved_conversation(
    monkeypatch,
    capsys,
):
    """
    History must only display messages belonging
    to the currently selected conversation.
    """

    queried_ids = []

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "History must not contact the inference server "
            "or save messages."
        )

    def load_active_messages(conversation_id):
        queried_ids.append(conversation_id)

        assert conversation_id == 21

        return [
            (
                "user",
                "Question only in conversation 21",
            ),
            (
                "assistant",
                "Answer only in conversation 21",
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
        lambda value: value == 21,
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
            "21",
            "history",
            "history 1",
            "exit",
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

    output = capsys.readouterr().out

    assert "Question only in conversation 21" in output
    assert "Answer only in conversation 21" in output

    assert queried_ids == [21]


def test_private_history_does_not_query_saved_messages(
    monkeypatch,
    capsys,
):
    """Private history must never retrieve persisted messages."""

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Private history must not access saved messages "
            "or persist conversation data."
        )

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
        "load_messages",
        forbidden,
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
            "p",
            "history",
            "exit",
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

    output = capsys.readouterr().out

    assert "current terminal scrollback" in output
