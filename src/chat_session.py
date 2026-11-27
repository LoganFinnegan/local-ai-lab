"""Track whether a conversation should be persisted."""

from database import create_conversation, save_message

class ChatSession:
    """Keep private sessions out of SQLite while preserving normal saves."""

    def __init__(self, conversation_id=None, private=False):
        if private and conversation_id is not None:
            raise ValueError(
                "A private session cannot resume a saved conversation."
            )

        self.conversation_id = conversation_id
        self.private = private

    def record_message(self, role, content, title=None):
        """Save a message unless this is a private session."""

        if self.private:
            return

        if self.conversation_id is None:
            if title is None:
                raise ValueError(
                    "A new saved conversation requires a title."
                )

            self.conversation_id = create_conversation(title)

        save_message(self.conversation_id, role, content)
