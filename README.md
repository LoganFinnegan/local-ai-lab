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
rebuild Python message history
  ↓
send complete context to Qwen
```

The model itself is not being retrained and does not permanently remember the conversation.

Persistence comes from storing the conversation in SQLite and supplying that history to the model again on future requests.

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
│   ├── database.py
│   └── llm_client.py
├── workflows/
│   └── SDXL_Model_Comparison.json
├── .gitignore
└── README.md
```

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
- GPU monitoring scripts
- VS Code startup tasks

Planned:

- Automated tests
- Improved conversation navigation
- Context-window management
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
