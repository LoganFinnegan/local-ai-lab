# Local AI Lab

A hands-on local AI project focused on understanding how model inference, GPU acceleration, HTTP APIs, application state, persistence, containers, and local infrastructure fit together.

The project started with local image generation and has expanded into a persistent local LLM application running on consumer AMD hardware.

## Current Hardware

- AMD Radeon RX 9070 XT - 16 GB VRAM
- AMD Ryzen 7 9800X3D
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
Python application
  ↓
SQLite conversation history
```

The current LLM runs locally through `llama.cpp` using ROCm acceleration on the RX 9070 XT.

The server is exposed only on the local machine at:

```text
http://127.0.0.1:8080
```

The Python application communicates with the OpenAI-compatible llama.cpp API.

The Python application can currently be run either:

- directly through the local Python virtual environment
- inside a Docker container

The llama.cpp server, model, ROCm runtime, and GPU remain on the host machine rather than inside the application container.

### Containerized Application

The current Docker layout separates the application layer from the inference layer:

```text
Docker container
├── Python 3.12
├── application code
├── context management
├── database code
├── tests
└── Python dependencies
        │
        │ HTTP
        ↓
Ubuntu host
├── llama-server
├── Qwen3-14B
├── ROCm
└── AMD GPU
```

SQLite data is bind-mounted from the host so conversation history survives container deletion and recreation.

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

The Python application currently supports:

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
- Environment-based server and model configuration
- Clear startup errors when llama-server is unavailable
- Dockerized application environment
- Persistent SQLite storage across disposable Docker containers
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

The Python application reads the active context size from llama.cpp at runtime and reserves 4,096 tokens for model output.

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

## Configuration

Runtime configuration is handled through environment variables.

The main configurable values are:

```text
LLM_SERVER_URL
MODEL_PATH
```

The Python application provides local defaults where appropriate, while Docker Compose can load machine-specific values from a local `.env` file.

The repository includes:

```text
.env.example
```

as a template.

The real:

```text
.env
```

file is ignored by Git so machine-specific configuration and future secrets are not committed.

## First-Time Native Setup

This project uses a Python virtual environment for its Python dependencies when running directly on the host.

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

The Python virtual environment and Docker image do not install the local AI inference runtime itself.

The current development setup also requires:

- a compatible inference machine
- a working `llama.cpp` build
- a compatible model
- sufficient CPU/GPU memory for the selected model

The current development machine additionally uses:

- AMD GPU drivers
- ROCm
- AMD/HIP support in llama.cpp
- Qwen3-14B-Q4_K_M GGUF

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

Then open another terminal and start the application:

```bash
python src/llm_client.py
```

### Manual Native Startup

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

Then start the chat application:

```bash
python src/llm_client.py
```

### Docker Startup

Docker is optional. The application can still be run directly through Python.

The current Docker setup containerizes the Python application while leaving llama.cpp and GPU inference on the host machine.

First create a local environment file:

```bash
cp .env.example .env
```

Edit `.env` for the local machine.

Example:

```text
LLM_SERVER_URL=http://127.0.0.1:8080
MODEL_PATH=/path/to/your/model.gguf
```

Start the llama.cpp server separately on the host.

On the current development machine:

```bash
./scripts/start_llama_server.sh
```

Confirm the server is responding:

```bash
curl -sS http://127.0.0.1:8080/props
```

Build the application image:

```bash
docker compose build
```

Run the application:

```bash
docker compose run --rm app
```

The `--rm` option removes the disposable application container after it exits.

Conversation history remains available because:

```text
./data
```

on the host is mounted inside the container at:

```text
/app/data
```

The container can therefore be destroyed and recreated without deleting the SQLite conversation database.

### Current Docker Networking

The current Compose configuration uses:

```text
network_mode: host
```

on Linux.

This allows the containerized application to reach the host llama.cpp server through:

```text
http://127.0.0.1:8080
```

without exposing llama-server to the wider network.

This networking configuration has been tested on the current Ubuntu development machine.

The application image itself is not intended to depend on AMD hardware, but the current Compose networking and helper scripts are Linux-focused. Running the project with a different host environment, such as macOS, may require different inference-server and container-network configuration.

## Docker Files

The repository includes:

```text
Dockerfile
.dockerignore
compose.yaml
.env.example
```

Their roles are:

```text
Dockerfile
→ describes how the application image is built

.dockerignore
→ excludes unnecessary local files from the Docker build context

compose.yaml
→ describes how the application container is configured and run

.env.example
→ documents machine-specific environment variables
```

The Docker image currently contains:

- Python 3.12
- Python dependencies
- application source code
- pytest configuration
- automated tests

It does not contain:

- Qwen model files
- llama.cpp
- ROCm
- AMD drivers
- the host virtual environment
- Git history
- local conversation data

## Optional System Monitoring

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
├── .github/
│   └── workflows/
│       └── tests.yml
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
│   ├── config.py
│   ├── context_manager.py
│   ├── database.py
│   └── llm_client.py
├── tests/
│   ├── test_context_manager.py
│   └── test_database.py
├── workflows/
│   └── SDXL_Model_Comparison.json
├── .dockerignore
├── .env.example
├── .gitignore
├── compose.yaml
├── Dockerfile
├── pytest.ini
├── requirements.txt
└── README.md
```

## Testing

Automated tests currently cover database behavior and context-budgeting logic.

### Native Testing

Run the test suite from the repository root with the virtual environment active:

```bash
python -m pytest -v
```

### Docker Testing

The same test suite can be run inside the application container:

```bash
docker compose run --rm app python -m pytest -v
```

This verifies that the Python application and tests also work inside the reproducible container environment rather than only inside the host development environment.

The context-manager tests cover:

- returning the complete history when it fits
- removing the oldest conversation turns when necessary
- preserving the original full message history
- rejecting requests that cannot fit even after truncation
- efficient cutoff selection with a large synthetic conversation

The database tests cover:

- creating conversations
- saving and loading messages
- checking whether conversations exist
- deleting conversations and associated messages
- handling deletion of missing conversations
- pagination through conversation history
- ordering conversations by most recent activity

## Continuous Integration

GitHub Actions automatically runs the Python test suite on pushes and pull requests. The workflow can also be started manually from GitHub Actions.

The CI workflow:

- runs on Ubuntu 24.04
- uses Python 3.12
- installs dependencies from `requirements.txt`
- checks Python syntax
- runs the pytest suite
- uses read-only repository permissions
- has a five-minute job timeout

The CI workflow tests the Python database and context-management code.

It does not run the local llama.cpp/Qwen/ROCm inference stack because the standard GitHub-hosted runner does not provide the required local GPU environment.

## What This Project Is Exploring

The goal is not simply to run an AI model.

The project is being used to understand the systems surrounding local AI applications, including:

- Linux processes and filesystem behavior
- users, groups, and permissions
- AMD GPU compute and ROCm
- model loading and VRAM usage
- quantized GGUF models
- HTTP request/response flow
- streaming APIs
- client/server boundaries
- Python data structures
- application state
- relational databases
- SQL queries and relationships
- transactions and persistence
- context windows and token budgeting
- algorithmic scaling
- automated testing
- continuous integration
- environment-based configuration
- Docker images and containers
- bind-mounted persistent storage
- container networking
- reproducible application environments
- debugging across multiple software layers

## Current Development Status

Working:

- Local Qwen inference
- AMD GPU acceleration
- llama.cpp HTTP server
- Streaming Python application
- Multi-turn context
- SQLite persistence
- Multiple conversations
- Conversation resume
- Conversation deletion
- Automatic conversation titles
- Runtime context-window detection
- Token-aware context budgeting
- Recent-history context truncation
- Environment-based configuration
- Startup error handling
- Automated database tests
- Automated context-manager tests
- GPU monitoring scripts
- VS Code startup tasks
- GitHub Actions CI
- Automated test runs on pushes and pull requests
- Dockerized Python application
- Docker Compose runtime configuration
- Persistent SQLite storage across disposable containers
- Test execution inside Docker

Planned:

- Retrieval-Augmented Generation (RAG)
- Retrieval from private documents
- Source-aware retrieval and provenance
- Tool calling
- Optional web/search tools
- Improved long-term memory and context handling
- Private/non-persistent sessions
- Additional logging and observability
- More portable Docker networking for non-Linux hosts
- Additional application interfaces beyond the terminal client

## Privacy

The current LLM can run entirely on the local machine.

The llama.cpp server binds to:

```text
127.0.0.1
```

rather than exposing the service to the local network.

Conversation history is stored locally in SQLite and the database file is excluded from Git.

The Docker configuration bind-mounts local application data rather than embedding conversation history into the application image.

No cloud model API is required for the current local inference workflow.