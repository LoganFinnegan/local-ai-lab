# Local AI Lab

A hands-on local AI project focused on understanding how model inference, GPU acceleration, HTTP APIs, application state, persistence, and local infrastructure fit together.

The project started with local image generation and has expanded into a persistent local LLM client running entirely on consumer AMD hardware.

## Current Hardware

- AMD Radeon RX 9070 XT - 16 GB VRAM
- AMD Ryzen 7 9800
- 32 GB RAM
- Ubuntu Linux

## Current Stack

### Local LLM

```text
Ubuntu
  ↓
AMDGPU
  ↓
ROCm
  ↓
llama.cpp
  ↓
Qwen3-14B-Q4_K_M
  ↓
Local HTTP API
  ↓
Python client
  ↓
SQLite conversation history
```

The current LLM runs locally through `llama.cpp` using ROCm acceleration on the RX 9070 XT.

The server is exposed only on the local machine at:

```text
http://127.0.0.1:8080
```

The Python client communicates with the OpenAI-compatible `/v1/chat/completions` endpoint.

### Image Generation

Earlier experiments used:

```text
Ubuntu
  ↓
AMDGPU
  ↓
ROCm
  ↓
PyTorch
  ↓
ComfyUI
  ↓
SDXL models
```

Models tested include:

- SDXL Lightning
- Juggernaut XL v9

The repository includes a saved ComfyUI workflow for controlled model comparisons using the same prompt, resolution, and seed.

## Current LLM Features

The Python client currently supports:

- Streaming responses from llama.cpp
- Configurable display of Qwen reasoning output
- Multi-turn conversation context
- SQLite-backed persistent conversation history
- Resuming conversations after the Python process exits
- Multiple saved conversations
- Automatic conversation titles based on the first user message
- Listing recent conversations
- Loading older conversations
- Deleting conversations with confirmation
- Runtime context-window detection from llama.cpp
- Token-aware context budgeting
- Automatic truncation of older conversation turns when necessary
- Context-usage reporting during development
- Local-only inference with no required cloud API calls

Conversation history is stored locally in:

```text
data/chat_history.db
```

The database file is ignored by Git and is not committed to the repository.

## Conversation Persistence

The application stores conversations and messages separately.

```text
conversations
    │
    ├── conversation 1
    │       ├── user message
    │       ├── assistant message
    │       └── ...
    │
    └── conversation 2
            ├── user message
            └── assistant message
```

When an existing conversation is resumed:

```text
SQLite
  ↓
load saved messages
  ↓
rebuild complete Python message history
  ↓
calculate available input-token budget
  ↓
fit recent conversation history to that budget
  ↓
send the fitted request context to Qwen
```

The complete conversation remains stored in SQLite even when older messages are omitted from a particular model request.

The model itself is not being retrained and does not permanently remember the conversation.

Persistence comes from storing the conversation in SQLite and supplying the relevant conversation history to the model again on future requests.

## Context Budgeting

The llama.cpp server is currently configured with a 32,768-token context window.

The Python client reads the active context size from llama.cpp at runtime and reserves 4,096 tokens for model output.

Before inference, the application counts the tokens required by the current conversation using llama.cpp's tokenizer.

If the full conversation fits within the available input budget, it is sent unchanged.

If the conversation exceeds the input budget, the context manager:

- preserves the system message
- keeps the newest complete conversation history that fits
- begins retained history on a user message rather than an orphaned assistant response
- removes older messages only from the current model request
- leaves the complete conversation unchanged in SQLite

A binary search is used to locate the oldest conversation position that still fits within the available input budget rather than checking every possible cutoff one at a time.

Current context selection is based on recency rather than relevance. Retrieval of older relevant information is planned for later RAG work.

## First-Time Setup

This project uses a Python virtual environment for its Python dependencies.

From the repository root, create the virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

When the virtual environment is active, the terminal prompt should normally begin with:

```text
(.venv)
```

The virtual environment itself is local to the machine and is excluded from Git. `requirements.txt` records the Python packages needed to recreate it.

### External Requirements

The Python virtual environment does not install the local AI runtime itself.

The current development setup also requires:

- AMD GPU drivers
- ROCm
- a working `llama.cpp` build with AMD/HIP support
- the Qwen3-14B-Q4_K_M GGUF model

The current helper script expects:

```text
llama.cpp:
~/ai/llama.cpp

model:
~/ai/models/llm/Qwen3-14B-Q4_K_M.gguf
```

These paths can be changed in:

```text
scripts/start_llama_server.sh
```

## Running the LLM Environment

Two helper scripts are included:

```text
scripts/
├── monitor_gpu.sh
└── start_llama_server.sh
```

### VS Code

The repository includes VS Code Tasks for starting the local development environment.

Open the Command Palette:

```text
Ctrl+Shift+P
```

Select:

```text
Tasks: Run Task
```

Then run:

```text
Start Local AI Environment
```

This starts:

- llama.cpp server in a dedicated VS Code terminal
- AMD GPU monitoring in a dedicated VS Code terminal

Then open another terminal and start the client:

```bash
python src/llm_client.py
```

### Manual Startup

Activate the Python virtual environment if it is not already active:

```bash
source .venv/bin/activate
```

The same components can be started without VS Code.

From the repository root:

```bash
./scripts/start_llama_server.sh
```

In another terminal:

```bash
./scripts/monitor_gpu.sh
```

Then start the chat client:

```bash
python src/llm_client.py
```

### Optional System Monitoring

For general CPU, RAM, process, network, and disk monitoring:

```bash
gnome-system-monitor
```

On this development machine, GNOME System Monitor should be launched from a normal Ubuntu terminal rather than the Snap-installed VS Code integrated terminal because of a GSettings environment conflict.

## llama.cpp Configuration

The current server configuration uses:

```bash
./build/bin/llama-server \
  -m ~/ai/models/llm/Qwen3-14B-Q4_K_M.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --offline \
  --ctx-size 32768 \
  --parallel 1 \
  --fit on \
  --fit-target 3072
```

Current model:

```text
Qwen3-14B-Q4_K_M.gguf
```

The 32K context configuration provides significantly more usable conversation history while still leaving some GPU memory available for the desktop and other applications.

## GPU Monitoring

GPU utilization and VRAM use can be monitored with:

```bash
./scripts/monitor_gpu.sh
```

The underlying command is:

```bash
watch -n 1 rocm-smi --showmemuse --showuse
```

This makes it easier to observe the difference between:

- VRAM allocation
- GPU compute utilization
- idle model memory
- active inference load

## Project Structure

```text
local-ai-lab/
├── .vscode/
│   └── tasks.json
├── data/
│   └── chat_history.db        # local only, ignored by Git
├── docs/
│   ├── architecture.md
│   ├── experiments.md
│   └── troubleshooting.md
├── scripts/
│   ├── monitor_gpu.sh
│   └── start_llama_server.sh
├── src/
│   ├── context_manager.py
│   ├── database.py
│   └── llm_client.py
├── tests/
│   ├── test_context_manager.py
│   └── test_database.py
├── workflows/
│   └── SDXL_Model_Comparison.json
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

## Testing

Automated tests currently cover database behavior and context-budgeting logic.

Run the test suite from the repository root with the virtual environment active:

```bash
python -m pytest -v
```

The context-manager tests cover:

- returning the complete history when it fits
- removing the oldest conversation turns when necessary
- preserving the original full message history
- rejecting requests that cannot fit even after truncation
- efficient cutoff selection with a large synthetic conversation

The large-history test verifies that context selection does not require checking every possible starting position.

## What This Project Is Exploring

The goal is not simply to run an AI model.

The project is being used to understand the systems surrounding local AI applications, including:

- Linux processes and filesystem behavior
- AMD GPU compute and ROCm
- model loading and VRAM usage
- quantized GGUF models
- HTTP request/response flow
- streaming APIs
- Python data structures
- application state
- relational databases
- SQL queries and relationships
- persistence across process restarts
- context windows and token budgeting
- algorithmic scaling
- automated testing
- debugging across multiple software layers

## Current Development Status

Working:

- Local Qwen inference
- AMD GPU acceleration
- llama.cpp HTTP server
- Streaming Python client
- Multi-turn context
- SQLite persistence
- Multiple conversations
- Conversation resume
- Conversation deletion
- Automatic conversation titles
- Runtime context-window detection
- Token-aware context budgeting
- Recent-history context truncation
- Automated database tests
- Automated context-manager tests
- GPU monitoring scripts
- VS Code startup tasks

Planned:

- Improved conversation navigation
- Retrieval-Augmented Generation (RAG)
- Retrieval from private documents
- Tool calling
- Private/non-persistent sessions
- Docker and reproducible environment setup
- CI
- Additional logging and health checks

## Privacy

The current LLM can run entirely on the local machine.

The llama.cpp server binds to:

```text
127.0.0.1
```

rather than exposing the service to the local network.

Conversation history is stored locally in SQLite and the database file is excluded from Git.
