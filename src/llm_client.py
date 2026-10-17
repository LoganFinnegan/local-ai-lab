import json
import requests

from database import (
    conversation_exists,
    create_conversation,
    delete_conversation,
    initialize_database,
    list_conversations,
    load_messages,
    save_message,
)


URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "/home/logan/ai/models/llm/Qwen3-14B-Q4_K_M.gguf"

SHOW_REASONING = False

SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are a direct, technically capable assistant. "
        "Assume the user is competent. "
        "Answer the question without unnecessary praise, hand-holding, "
        "or offers of additional help. "
        "Be concise unless more detail is requested."
    ),
}


def make_title(prompt):
    title = prompt.strip()

    if len(title) <= 60:
        return title

    return title[:57].rstrip() + "..."


def choose_conversation():
    offset = 0
    limit = 5

    while True:
        conversations = list_conversations(
            limit=limit,
            offset=offset,
        )

        print("\nLocal AI\n")

        if conversations:
            if offset == 0:
                print("Recent conversations:")
            else:
                print("Older conversations:")

            for conversation in conversations:
                conversation_id, title, created_at, last_active = conversation

                print(
                    f"{conversation_id} - {title} "
                    f"({last_active})"
                )

            print()

        elif offset == 0:
            print("No saved conversations yet.\n")

        else:
            print("No more saved conversations.\n")
            offset = 0
            limit = 5
            continue

        print("Press Enter to start a new chat")
        print("Enter a conversation ID to resume")
        print("Enter D <ID> to delete a conversation")

        if conversations:
            print("Enter M to show more")

        choice = input("\n>>> ").strip()

        if choice == "":
            return None

        if choice.lower() == "m":
            offset += limit
            limit = 10
            continue

        if choice.lower().startswith("d "):
            delete_id_text = choice[2:].strip()

            try:
                delete_id = int(delete_id_text)
            except ValueError:
                print("\nPlease enter D followed by a conversation ID.")
                continue

            if not conversation_exists(delete_id):
                print(f"\nConversation {delete_id} does not exist.")
                continue

            confirmation = input(
                f"\nDelete conversation {delete_id} "
                "and all of its messages? [y/N]: "
            ).strip().lower()

            if confirmation == "y":
                delete_conversation(delete_id)
                print(f"\nDeleted conversation {delete_id}.")

                offset = 0
                limit = 5

            else:
                print("\nDelete cancelled.")

            continue

        try:
            conversation_id = int(choice)
        except ValueError:
            print(
                "\nPlease enter a conversation ID, "
                "D <ID>, M, or press Enter."
            )
            continue

        if not conversation_exists(conversation_id):
            print(f"\nConversation {conversation_id} does not exist.")
            continue

        return conversation_id


initialize_database()

conversation_id = choose_conversation()

messages = [SYSTEM_MESSAGE]

if conversation_id is not None:
    stored_messages = load_messages(conversation_id)

    for role, content in stored_messages:
        messages.append({
            "role": role,
            "content": content,
        })

    print(f"\nResumed conversation {conversation_id}.")

else:
    print("\nStarted a new conversation.")


while True:
    prompt = input("\n>>> ")

    if prompt.strip().lower() in ["exit", "quit"]:
        break

    if conversation_id is None:
        title = make_title(prompt)
        conversation_id = create_conversation(title)

    user_message = {
        "role": "user",
        "content": prompt,
    }

    messages.append(user_message)

    save_message(
        conversation_id,
        "user",
        prompt,
    )

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
    }

    try:
        response = requests.post(
            URL,
            json=payload,
            timeout=120,
            stream=True,
        )

        response.raise_for_status()

    except requests.exceptions.ConnectionError:
        print("Could not connect to llama-server.")
        raise SystemExit(1)

    except requests.exceptions.Timeout:
        print("The request timed out.")
        raise SystemExit(1)

    except requests.exceptions.HTTPError as error:
        print(f"The server returned an HTTP error: {error}")
        raise SystemExit(1)

    print()

    showing_reasoning = False
    showing_answer = False
    assistant_answer = ""

    for line in response.iter_lines():
        if not line:
            continue

        line = line.decode("utf-8")

        if not line.startswith("data: "):
            continue

        data_text = line.removeprefix("data: ")

        if data_text == "[DONE]":
            break

        data = json.loads(data_text)

        delta = data["choices"][0]["delta"]

        reasoning = delta.get("reasoning_content") or ""
        content = delta.get("content") or ""

        if SHOW_REASONING and reasoning:
            if not showing_reasoning:
                print("[Reasoning]")
                showing_reasoning = True

            print(reasoning, end="", flush=True)

        if content:
            if not showing_answer:
                print("\n[Answer]")
                showing_answer = True

            print(content, end="", flush=True)
            assistant_answer += content

    print()

    assistant_message = {
        "role": "assistant",
        "content": assistant_answer,
    }

    messages.append(assistant_message)

    save_message(
        conversation_id,
        "assistant",
        assistant_answer,
    )
    