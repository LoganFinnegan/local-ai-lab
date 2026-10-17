import sqlite3
from pathlib import Path

import database


def setup_test_database(tmp_path):
    database.DB_PATH = Path(tmp_path) / "test_chat_history.db"
    database.initialize_database()


def test_create_and_load_messages(tmp_path):
    setup_test_database(tmp_path)

    conversation_id = database.create_conversation("Test conversation")

    database.save_message(
        conversation_id,
        "user",
        "Hello",
    )

    database.save_message(
        conversation_id,
        "assistant",
        "Hi there",
    )

    messages = database.load_messages(conversation_id)

    assert messages == [
        ("user", "Hello"),
        ("assistant", "Hi there"),
    ]


def test_conversation_exists(tmp_path):
    setup_test_database(tmp_path)

    conversation_id = database.create_conversation("Exists test")
    missing_conversation_id = 9999

    assert database.conversation_exists(conversation_id) is True
    assert database.conversation_exists(missing_conversation_id) is False


def test_delete_conversation(tmp_path):
    setup_test_database(tmp_path)

    conversation_id = database.create_conversation("Delete test")

    database.save_message(
        conversation_id,
        "user",
        "Delete me",
    )

    deleted = database.delete_conversation(conversation_id)

    assert deleted is True
    assert database.conversation_exists(conversation_id) is False
    assert database.load_messages(conversation_id) == []


def test_delete_missing_conversation(tmp_path):
    setup_test_database(tmp_path)

    deleted = database.delete_conversation(9999)

    assert deleted is False


def test_list_conversations_respects_limit_and_offset(tmp_path):
    setup_test_database(tmp_path)

    conversation_ids = []

    for number in range(6):
        conversation_id = database.create_conversation(
            f"Conversation {number + 1}"
        )
        conversation_ids.append(conversation_id)

    first_page = database.list_conversations(
        limit=5,
        offset=0,
    )

    second_page = database.list_conversations(
        limit=5,
        offset=5,
    )

    assert len(first_page) == 5
    assert len(second_page) == 1

    assert [conversation[0] for conversation in first_page] == list(
        reversed(conversation_ids[1:])
    )

    assert second_page[0][0] == conversation_ids[0]


def test_list_conversations_orders_by_last_activity(tmp_path):
    setup_test_database(tmp_path)

    older_conversation_id = database.create_conversation(
        "Older conversation"
    )
    newer_conversation_id = database.create_conversation(
        "Newer conversation"
    )

    database.save_message(
        older_conversation_id,
        "user",
        "Make the older conversation active again",
    )

    database.save_message(
        newer_conversation_id,
        "user",
        "Newer conversation message",
    )

    connection = sqlite3.connect(database.DB_PATH)

    connection.execute(
        """
        UPDATE messages
        SET created_at = ?
        WHERE conversation_id = ?
        """,
        ("2026-10-17 13:00:00", newer_conversation_id),
    )

    connection.execute(
        """
        UPDATE messages
        SET created_at = ?
        WHERE conversation_id = ?
        """,
        ("2026-10-17 14:00:00", older_conversation_id),
    )

    connection.commit()
    connection.close()

    conversations = database.list_conversations()

    conversation_ids = [
        conversation[0]
        for conversation in conversations
    ]

    assert conversation_ids[0] == older_conversation_id
    assert newer_conversation_id in conversation_ids
