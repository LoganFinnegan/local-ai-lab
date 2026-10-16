import json
import requests

URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "/home/logan/ai/models/llm/Qwen3-14B-Q4_K_M.gguf"

prompt = input("Ask Qwen: ")

payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": prompt,
        }
    ],
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

    if reasoning:
        if not showing_reasoning:
            print("[Reasoning]")
            showing_reasoning = True

        print(reasoning, end="", flush=True)

    if content:
        if not showing_answer:
            print("\n\n[Answer]")
            showing_answer = True

        print(content, end="", flush=True)

print()
