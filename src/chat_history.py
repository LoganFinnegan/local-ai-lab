"""Display messages from the current conversation only."""


HISTORY_USAGE = (
    "Usage: history [N] "
    "(N must be a positive whole number)."
)


def history_response(command, messages, private=False):
    """
    Handle history commands for the active conversation.

    Return None if this is not a history command.
    Never display system messages.
    """

    parts = command.strip().split()

    if not parts or parts[0].lower() != "history":
        return None

    if len(parts) > 2:
        return HISTORY_USAGE

    limit = None

    if len(parts) == 2:
        count_text = parts[1]

        if (
            not count_text.isascii()
            or not count_text.isdecimal()
        ):
            return HISTORY_USAGE

        try:
            limit = int(count_text)
        except ValueError:
            return HISTORY_USAGE

        if limit <= 0:
            return HISTORY_USAGE

    if private:
        return (
            "Private chats are not saved. "
            "Use the current terminal scrollback "
            "to review this conversation."
        )

    entries = [
        (message["role"], message["content"])
        for message in messages
        if message["role"] in ("user", "assistant")
    ]

    if limit is not None:
        entries = entries[-limit:]

    if not entries:
        return "No messages in this conversation yet."

    display = [
        (
            f"{'You' if role == 'user' else 'Assistant'}:"
            f"\n{content}"
        )
        for role, content in entries
    ]

    return "[History]\n\n" + "\n\n".join(display)
