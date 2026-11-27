import pytest

import chat_session

def test_private_session_never_calls_database(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Private sessions must not access storage.")

    monkeypatch.setattr(chat_session, "create_conversation", forbidden)
    monkeypatch.setattr(chat_session, "save_message", forbidden)

    session = chat_session.ChatSession(private=True)
    session.record_message("user", "Private question", title="Private question")
    session.record_message("assistant", "Private answer")

    assert session.conversation_id is None

def test_new_saved_session_creates_conversation_once(monkeypatch):
    titles = []
    saved = []

    def create(title):
        titles.append(title)
        return 17

    def save(conversation_id, role, content):
        saved.append((conversation_id, role, content))

    monkeypatch.setattr(chat_session, "create_conversation", create)
    monkeypatch.setattr(chat_session, "save_message", save)

    session = chat_session.ChatSession()
    session.record_message("user", "Hello", title="Hello")
    session.record_message("assistant", "Hi")

    assert session.conversation_id == 17
    assert titles == ["Hello"]
    assert saved == [
        (17, "user", "Hello"),
        (17, "assistant", "Hi"),
    ]

def test_resumed_session_reuses_existing_conversation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Resuming must not create another conversation.")

    saved = []
    monkeypatch.setattr(chat_session, "create_conversation", forbidden)
    monkeypatch.setattr(
        chat_session,
        "save_message",
        lambda conversation_id, role, content: saved.append(
            (conversation_id, role, content)
        ),
    )

    session = chat_session.ChatSession(conversation_id=42)
    session.record_message("user", "Another question")

    assert saved == [(42, "user", "Another question")]

def test_private_session_cannot_resume_saved_conversation():
    with pytest.raises(ValueError, match="cannot resume"):
        chat_session.ChatSession(conversation_id=42, private=True)

def test_new_saved_session_requires_title(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Missing title must fail before database access.")

    monkeypatch.setattr(chat_session, "create_conversation", forbidden)

    session = chat_session.ChatSession()

    with pytest.raises(ValueError, match="requires a title"):
        session.record_message("user", "Hello")
