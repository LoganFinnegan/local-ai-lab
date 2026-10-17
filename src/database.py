import sqlite3
from pathlib import Path


DB_PATH = Path("data/chat_history.db")


def initialize_database():
    DB_PATH.parent.mkdir(exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON")

    connection.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id)
                REFERENCES conversations(id)
        )
    """)

    connection.commit()
    connection.close()


def create_conversation(title):
    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON")

    cursor = connection.execute(
        "INSERT INTO conversations (title) VALUES (?)",
        (title,),
    )

    conversation_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return conversation_id


def save_message(conversation_id, role, content):
    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON")

    connection.execute(
        """
        INSERT INTO messages (conversation_id, role, content)
        VALUES (?, ?, ?)
        """,
        (conversation_id, role, content),
    )

    connection.commit()
    connection.close()


def load_messages(conversation_id):
    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON")

    cursor = connection.execute(
        """
        SELECT role, content
        FROM messages
        WHERE conversation_id = ?
        ORDER BY id
        """,
        (conversation_id,),
    )

    messages = cursor.fetchall()

    connection.close()

    return messages


def list_conversations(limit=5, offset=0):
    connection = sqlite3.connect(DB_PATH)

    cursor = connection.execute(
        """
        SELECT
            conversations.id,
            conversations.title,
            conversations.created_at,
            COALESCE(
                MAX(messages.created_at),
                conversations.created_at
            ) AS last_active
        FROM conversations
        LEFT JOIN messages
            ON messages.conversation_id = conversations.id
        GROUP BY conversations.id
        ORDER BY last_active DESC, conversations.id DESC
        LIMIT ?
        OFFSET ?
        """,
        (limit, offset),
    )

    conversations = cursor.fetchall()

    connection.close()

    return conversations


def conversation_exists(conversation_id):
    connection = sqlite3.connect(DB_PATH)

    cursor = connection.execute(
        """
        SELECT id
        FROM conversations
        WHERE id = ?
        """,
        (conversation_id,),
    )

    conversation = cursor.fetchone()

    connection.close()

    return conversation is not None


def delete_conversation(conversation_id):
    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON")

    try:
        connection.execute(
            """
            DELETE FROM messages
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        )

        cursor = connection.execute(
            """
            DELETE FROM conversations
            WHERE id = ?
            """,
            (conversation_id,),
        )

        connection.commit()

        return cursor.rowcount > 0

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


if __name__ == "__main__":
    initialize_database()
    print(f"Database initialized at {DB_PATH}")
    