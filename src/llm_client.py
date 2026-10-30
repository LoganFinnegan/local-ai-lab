import json

import requests

import config

from context_manager import fit_messages_to_budget

from database import (
    conversation_exists,
    create_conversation,
    delete_conversation,
    initialize_database,
    list_conversations,
    load_messages,
    save_message,
)


MAX_OUTPUT_TOKENS = 4096

# Temporary development override.
# Set this to None before the feature is finished.
DEVELOPMENT_INPUT_TOKEN_BUDGET = None

SHOW_REASONING = False
SHOW_CONTEXT_USAGE = True

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


def get_context_window_tokens():
    response = requests.get(
        config.PROPS_URL,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    return data["default_generation_settings"]["n_ctx"]


def count_input_tokens(messages):
    payload = {
        "model": config.MODEL_PATH,
        "messages": messages,
    }

    response = requests.post(
        config.TOKEN_COUNT_URL,
        json=payload,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    return data["input_tokens"]


def make_title(prompt):
    title = prompt.strip()

    if len(title) <= 60:
        return title

    return title[:57].rstrip() + "..."


def choose_conversation():
    initial_limit = 5
    load_more_limit = 10

    displayed_conversations = list_conversations(
        limit=initial_limit,
        offset=0,
    )

    while True:
        print("\nLocal AI\n")

        if displayed_conversations:
            print("Recent conversations:")

            for conversation in displayed_conversations:
                conversation_id, title, created_at, last_active = conversation

                print(
                    f"{conversation_id} - {title} "
                    f"({last_active})"
                )

            print()

        else:
            print("No saved conversations yet.\n")

        has_more = bool(
            list_conversations(
                limit=1,
                offset=len(displayed_conversations),
            )
        )

        print("Press Enter to start a new chat")
        print("Enter a conversation ID to resume")
        print("Enter D <ID> to delete a conversation")

        if has_more:
            print("Enter M to show more")

        choice = input("\n>>> ").strip()

        if choice == "":
            return None

        if choice.lower() == "m":
            if not has_more:
                print("\nNo more saved conversations.")
                continue

            more_conversations = list_conversations(
                limit=load_more_limit,
                offset=len(displayed_conversations),
            )

            displayed_conversations.extend(more_conversations)
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

                displayed_conversations = list_conversations(
                    limit=initial_limit,
                    offset=0,
                )

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

context_window_tokens = get_context_window_tokens()

if context_window_tokens <= MAX_OUTPUT_TOKENS:
    raise SystemExit(
        "Invalid context configuration: "
        f"llama-server has a {context_window_tokens:,}-token "
        f"context window, but the client reserves "
        f"{MAX_OUTPUT_TOKENS:,} tokens for output."
    )

normal_input_token_budget = (
    context_window_tokens - MAX_OUTPUT_TOKENS
)

if DEVELOPMENT_INPUT_TOKEN_BUDGET is not None:
    input_token_budget = min(
        normal_input_token_budget,
        DEVELOPMENT_INPUT_TOKEN_BUDGET,
    )

    print(
        "\nDevelopment context-budget override enabled: "
        f"{input_token_budget:,} tokens"
    )

else:
    input_token_budget = normal_input_token_budget


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

    user_message = {
        "role": "user",
        "content": prompt,
    }

    messages.append(user_message)

    try:
        (
            request_messages,
            full_input_tokens,
            fitted_input_tokens,
            omitted_messages,
        ) = fit_messages_to_budget(
            messages,
            input_token_budget,
            count_input_tokens,
        )

    except ValueError as error:
        messages.pop()
        print(f"\nCould not send message: {error}")
        continue

    if conversation_id is None:
        title = make_title(prompt)
        conversation_id = create_conversation(title)

    save_message(
        conversation_id,
        "user",
        prompt,
    )

    if SHOW_CONTEXT_USAGE:
        if omitted_messages:
            omitted_tokens = (
                full_input_tokens - fitted_input_tokens
            )

            print(
                f"\nFull conversation input: "
                f"{full_input_tokens:,} tokens"
            )

            print(
                f"Context sent: "
                f"{fitted_input_tokens:,} / "
                f"{input_token_budget:,} tokens"
            )

            print(
                f"Older context omitted: "
                f"{omitted_tokens:,} tokens across "
                f"{omitted_messages} messages"
            )

        else:
            print(
                f"\nContext input: "
                f"{fitted_input_tokens:,} / "
                f"{input_token_budget:,} tokens"
            )

    payload = {
        "model": config.MODEL_PATH,
        "messages": request_messages,
        "stream": True,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }

    try:
        response = requests.post(
            config.CHAT_COMPLETIONS_URL,
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
