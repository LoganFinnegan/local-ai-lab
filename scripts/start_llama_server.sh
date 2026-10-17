#!/usr/bin/env bash

LLAMA_DIR="$HOME/ai/llama.cpp"
MODEL="$HOME/ai/models/llm/Qwen3-14B-Q4_K_M.gguf"

cd "$LLAMA_DIR" || exit 1

./build/bin/llama-server \
  -m "$MODEL" \
  --host 127.0.0.1 \
  --port 8080 \
  --offline \
  --ctx-size 32768 \
  --parallel 1 \
  --fit on \
  --fit-target 3072
  