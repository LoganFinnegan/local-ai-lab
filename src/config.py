import os

LLM_SERVER_URL = os.getenv(
    "LLM_SERVER_URL",
    "http://127.0.0.1:8080",
)

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.expanduser("~/ai/models/llm/Qwen3-14B-Q4_K_M.gguf"),
)

CHAT_COMPLETIONS_URL = f"{LLM_SERVER_URL}/v1/chat/completions"

TOKEN_COUNT_URL = f"{LLM_SERVER_URL}/v1/chat/completions/input_tokens"

PROPS_URL = f"{LLM_SERVER_URL}/props"
