
# Local AI Lab

A local-first AI engineering project exploring how model inference, GPU acceleration, HTTP APIs, application state, databases, containers, and Retrieval-Augmented Generation (RAG) fit together.

The project runs a quantized Qwen language model locally on consumer AMD hardware. A Python application communicates with llama.cpp over HTTP, manages persistent conversation history in SQLite, and provides a separate RAG pipeline for asking questions about local documents.

The goal is to understand and build the infrastructure surrounding AI applications—not simply download a model and run inference.

## Current Capabilities

### Local LLM Application

- Qwen3-14B inference through llama.cpp
- AMD GPU acceleration using ROCm
- OpenAI-compatible local HTTP API
- Streaming responses
- Multi-turn conversations
- Persistent SQLite conversation history
- Multiple saved conversations
- Conversation resume and deletion
- Automatic conversation titles
- Runtime context-window detection
- Token-aware context budgeting
- Automatic truncation of older conversation turns
- Configurable reasoning-output visibility
- Environment-based configuration
- Startup and connection error handling
- Dockerized Python application
- Persistent application data through bind mounts

### Retrieval-Augmented Generation

- Local text-document ingestion
- FastEmbed sentence embeddings
- Persistent Chroma vector storage
- SHA-256 document change detection
- Incremental indexing
- Detection of added, modified, and deleted documents
- Fixed-size overlapping text chunks
- Semantic similarity search
- Source and chunk metadata
- Retrieval inspection tools
- Qwen-generated answers grounded in retrieved passages
- Repeatable retrieval evaluation
- Automated, heuristic answer checks
- Timestamped evaluation reports

### Testing and Infrastructure

- Python unit tests with pytest
- GitHub Actions continuous integration
- Docker-based test execution
- GPU monitoring scripts
- VS Code startup tasks
- Local model and embedding-cache persistence

The latest local test run completed with **26 passing unit tests**.

## Hardware

The primary development machine uses:

| Component | Hardware |
|---|---|
| GPU | AMD Radeon RX 9070 XT, 16 GB VRAM |
| CPU | AMD Ryzen 7 9800X3D |
| System memory | 32 GB DDR5 |
| Operating system | Ubuntu Linux |
| Python | 3.12 |
| Local LLM | Qwen3-14B-Q4_K_M |

The current inference configuration is optimized for this machine. Other hardware may require different model sizes, GPU settings, or runtime configuration.

## Technology Stack

### Language Model

- **llama.cpp:** Local model inference server
- **Qwen3-14B-Q4_K_M:** Quantized language model
- **ROCm:** AMD GPU compute platform
- **HTTP:** Communication between the Python application and inference server
- **Python:** Application and integration logic

### Retrieval

- **FastEmbed:** Text embedding generation
- **BAAI/bge-small-en-v1.5:** Embedding model
- **Chroma:** Persistent vector database
- **SHA-256:** Document fingerprinting for incremental indexing

### Application Infrastructure

- **SQLite:** Persistent conversation history
- **Docker:** Containerized Python application
- **Docker Compose:** Container configuration and startup
- **pytest:** Automated testing
- **GitHub Actions:** Continuous integration

## Architecture

The application separates local model inference from Python application logic.

### LLM Application

```text
Python application
       |
       | HTTP
       v
llama.cpp server
       |
       v
Qwen3-14B
       |
       v
ROCm / AMD GPU
```

Conversation persistence operates independently:

```text
Python application
       |
       v
SQLite database
       |
       v
Saved conversations and messages
```

The llama.cpp server is a separate process. Stopping the Python application does not stop the inference server.

### RAG Pipeline

```text
Local text documents
       |
       v
Document loading and SHA-256 hashing
       |
       v
250-word chunks with 50-word overlap
       |
       v
FastEmbed embeddings
       |
       v
Persistent Chroma collection
       |
       v
Semantic similarity search
       |
       v
Retrieved document passages
       |
       v
Prompt construction
       |
       v
llama.cpp / Qwen
       |
       v
Generated answer
```

The embedding and inference models serve different purposes.

FastEmbed converts text into numerical vectors representing semantic relationships.

Chroma stores those vectors and searches for similar passages.

Qwen receives the retrieved text as reference material and generates a natural-language answer.

The RAG demonstrations currently run separately from the persistent conversational chat client.

## Conversation Persistence

The application uses SQLite to store conversations and messages.

```text
conversations
    |
    |-- conversation 1
    |      |-- user message
    |      |-- assistant message
    |      `-- ...
    |
    `-- conversation 2
           |-- user message
           `-- assistant message
```

The database is stored locally:

```text
data/chat_history.db
```

The file is excluded from Git.

When a conversation is resumed, the application:

1. Loads the saved messages from SQLite.
2. Reconstructs the conversation history in Python.
3. Determines the available model context budget.
4. Selects the newest complete conversation history that fits.
5. Sends that context to llama.cpp.

The entire history remains in SQLite even when older messages are excluded from an individual model request.

Qwen does not permanently remember conversations between requests. Conversation continuity is provided by the application.

## Context Management

The current llama.cpp configuration uses a 32,768-token context window.

The Python application reads the active context size from llama.cpp and reserves 4,096 tokens for model output.

Before inference, the application counts the input tokens using llama.cpp's tokenizer.

If the entire conversation fits, it is included.

If it exceeds the available input budget, the context manager:

- Preserves the system message
- Keeps the newest complete conversation history that fits
- Avoids starting retained history with an orphaned assistant response
- Excludes older messages only from the current request
- Leaves the original SQLite history unchanged

A binary search identifies the oldest conversation position that fits the available budget.

This avoids testing every possible cutoff sequentially.

Context selection is currently based on recency rather than relevance.

The RAG system demonstrates a separate mechanism for retrieving relevant information from stored documents.

## Retrieval-Augmented Generation

RAG allows the application to answer questions using external reference material without retraining the language model.

### Document Loading

Knowledge files are stored in:

```text
knowledge/
```

The current loader processes UTF-8 `.txt` files.

Example documents include:

```text
knowledge/
├── ai_notes.txt
├── docker_notes.txt
├── dune.txt
└── frankenstein.txt
```

These documents provide a small technical knowledge base alongside larger literary test documents.

### Text Chunking

Rather than embedding entire documents, the ingestion pipeline divides them into smaller chunks.

The current configuration uses:

```text
Chunk size: 250 words
Overlap:    50 words
```

Adjacent chunks share 50 words.

Overlap helps preserve context when information crosses a chunk boundary.

The original paragraph-based strategy produced many extremely small chunks when tested against a large document.

The fixed-size strategy provides more consistent chunk sizes.

### Embedding Generation

Text embeddings are generated with:

```text
BAAI/bge-small-en-v1.5
```

through FastEmbed.

The embedding-model cache is stored in:

```text
data/fastembed_cache/
```

The cache persists across application runs.

The embedding model may need to be downloaded on first use. Subsequent runs can reuse the cached files.

### Chroma Vector Storage

Chroma stores document embeddings and metadata persistently:

```text
data/chroma/
```

Each indexed chunk includes information such as:

- Source filename
- Document hash
- Chunk index
- Chunk text

Chroma uses cosine-distance-based similarity search.

The application uses Chroma's embedded persistent client. No separate Chroma server needs to be started.

### Incremental Indexing

Each document receives a SHA-256 hash calculated from its file contents.

During synchronization, the application compares the current knowledge directory against documents already indexed in Chroma.

Files can be classified as:

| Status | Action |
|---|---|
| Added | Generate and store embeddings |
| Changed | Replace the document's indexed chunks |
| Deleted | Remove the document's indexed chunks |
| Unchanged | Reuse existing embeddings |

This avoids regenerating embeddings for documents that have not changed.

**Important limitation:** File hashes detect changes to document contents, not changes to the chunking algorithm or embedding model. Changing those settings may require rebuilding the existing vector index.

### Retrieval

When a question is submitted:

1. FastEmbed generates an embedding for the question.
2. Chroma searches for similar stored document chunks.
3. The most relevant chunks are returned with similarity scores.
4. The application constructs a prompt containing the retrieved reference material.
5. Qwen generates an answer using that context.

Retrieval results preserve source filenames and chunk indices for inspection.

The current system does not yet provide fully verified, passage-level citations in generated answers.

## Large-Document Validation

The project uses *Frankenstein* by Mary Shelley as a realistic RAG test document.

The novel provides a substantially larger and more complex test than short notes or synthetic examples.

The text is available through [Project Gutenberg](https://www.gutenberg.org/ebooks/84).

### Initial Findings

The original paragraph-based chunking strategy produced:

```text
926 chunks
165 chunks under 20 words
```

Line-ending differences were also found to affect paragraph detection.

The ingestion process was updated to handle the source text consistently, and chunking was changed to fixed-size overlapping word windows.

### Updated Chunk Statistics

After re-indexing the knowledge base:

```text
Documents:              4
Chunks:                 408
Smallest chunk:         113 words
Largest chunk:          250 words
Average chunk:          248.9 words
Chunks under 20 words:  0
Chunks over 500 words:  0
```

These statistics describe the November 2026 validation dataset.

The result is a more consistent chunk distribution, although consistent chunk sizes do not automatically guarantee better retrieval relevance.

## Running the Application

### External Requirements

The Python virtual environment and Docker image do not install the local inference runtime.

The current development environment requires:

- A compatible llama.cpp build
- A compatible GGUF model
- Sufficient CPU/GPU memory
- AMD GPU drivers and ROCm for the current GPU setup

The helper script currently expects:

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

### Python Environment

From the repository root:

```bash
python -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Start llama.cpp

The server can be launched manually:

```bash
./scripts/start_llama_server.sh
```

Or through VS Code:

1. Open the Command Palette with `Ctrl+Shift+P`.
2. Select `Tasks: Run Task`.
3. Select `Start Local AI Environment`.

The VS Code task starts the llama.cpp server and GPU monitoring in separate terminals.

Confirm that the inference server is available:

```bash
curl -sS http://127.0.0.1:8080/props
```

The server listens locally at:

```text
http://127.0.0.1:8080
```

### Start the Conversation Client

With llama.cpp running:

```bash
python src/llm_client.py
```

The application can create, resume, and delete saved conversations.

### Inspect Document Chunking

```bash
python src/chunk_inspect.py
```

This reports document counts, chunk counts, and chunk-size statistics.

The embedding model and llama.cpp server are not required to inspect chunk sizes.

### Run Retrieval Only

```bash
python src/retrieval_demo.py
```

This loads the knowledge documents, synchronizes Chroma, and allows interactive semantic searches.

The llama.cpp server is not required.

### Run RAG Question Answering

With llama.cpp running:

```bash
python src/rag_demo.py
```

This performs retrieval and sends the retrieved reference material to Qwen for answer generation.

The response is generated by the local model rather than retrieved directly from the vector database.

## RAG Evaluation

The project includes a repeatable evaluation workflow.

Evaluation cases are stored in:

```text
evals/frankenstein.json
```

The current evaluation set contains six questions covering:

1. William Frankenstein's murder accusation
2. The creature's request for a female companion
3. The books discovered and read by the creature
4. Justine Moritz's conviction and execution
5. Justine's false confession
6. An intentionally unanswerable question

Each case can define:

- A question
- An expected source document
- Required evidence phrases
- Required answer phrases
- Phrases that should trigger manual review

### Retrieval Evaluation

Run:

```bash
python src/evaluate_rag.py --retrieval-only
```

This tests the real FastEmbed and Chroma retrieval pipeline without calling Qwen.

The evaluator checks whether expected evidence appears within the retrieved chunks.

The default retrieval limit is 10.

For the current Frankenstein test set, expected evidence was found at these ranks:

| Question | Evidence rank |
|---|---:|
| William's murder accusation | 5 |
| Female companion | 3 |
| Creature's books | 10 |
| Justine's verdict | 5 |
| Justine's false confession | 6 |
| Unanswerable question | N/A |

The unanswerable case intentionally has no expected evidence passage.

The books question remains a retrieval weakness: the relevant passage was only found at rank 10.

### Full RAG Evaluation

With llama.cpp running:

```bash
python src/evaluate_rag.py
```

This performs retrieval, generates answers with Qwen, and evaluates the results.

The evaluator checks for required phrases and flags selected misleading statements.

Reports are written to:

```text
data/rag_evaluations/
```

Reports use timestamped JSON filenames and contain:

- Evaluation metadata
- Retrieval results and similarity scores
- Retrieved chunk text
- Generated answers
- Evidence-check results
- Answer-check results
- Inference timing

Generated evaluation reports are excluded from Git.

### Evaluation Limitations

The answer checks are heuristic.

A passing answer means the required expressions were found and no configured review phrases were detected.

It does **not** prove that every statement in the answer is factually correct.

Development revealed both kinds of evaluation error:

**False positive:** An answer contained the expected facts but also made an unsupported claim. The keyword checker passed it.

**False negative:** Qwen correctly explained that information was unavailable, but used wording the evaluator did not recognize.

These findings are part of the evaluation process rather than evidence that keyword matching can provide complete factual verification.

The saved reports make it possible to inspect actual model responses instead of relying entirely on automated PASS/FAIL labels.

For meaningful changes to prompts, embeddings, retrieval settings, or chunking, generated answers should be reviewed alongside the automated results.

## Docker

Docker is optional for native development.

The current Docker setup packages the Python application separately from llama.cpp.

```text
Docker container
├── Python 3.12
├── application source
├── knowledge files
├── context management
├── database integration
├── tests
└── Python dependencies

Ubuntu host
├── llama-server
├── Qwen3-14B
├── ROCm
└── AMD GPU
```

The model, inference server, GPU drivers, and ROCm runtime remain on the host.

### Configuration

A local environment file can be created with:

```bash
cp .env.example .env
```

Example values:

```text
LLM_SERVER_URL=http://127.0.0.1:8080
MODEL_PATH=/path/to/your/model.gguf
```

The actual `.env` file is excluded from Git.

### Build the Application

```bash
docker compose build
```

### Run the Application

Start llama.cpp separately on the host, then run:

```bash
docker compose run --rm app
```

The `--rm` option removes the disposable container after the application exits.

### Persistent Data

Docker Compose bind-mounts:

```text
./data
```

from the host to:

```text
/app/data
```

inside the container.

This preserves local application data when containers are recreated.

### Container Networking

The current Linux configuration uses:

```text
network_mode: host
```

The container can therefore communicate with the host inference server at:

```text
http://127.0.0.1:8080
```

The networking configuration has been tested on the Ubuntu development machine.

Other operating systems may require different container networking.

### Docker Build Context

The `.dockerignore` file excludes unnecessary local development files, including:

- Git metadata
- Python virtual environments
- Local application databases
- Chroma storage
- FastEmbed caches
- Environment secrets
- Large model files

The Dockerfile copies the application source, knowledge files, and test files into the image.

The current Docker image does not copy the `evals/` directory, so the documented evaluation commands are intended for native execution.

## llama.cpp Configuration

The current development configuration is equivalent to:

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

The current configuration limits parallelism and uses llama.cpp memory-fitting options to leave GPU resources available for the desktop.

Earlier configurations consumed excessive VRAM and caused instability in other applications.

The current configuration has been more stable on the development machine.

GPU memory availability and inference performance depend on the model, context size, runtime, and other active applications.

## GPU Monitoring

Run:

```bash
./scripts/monitor_gpu.sh
```

The script uses:

```bash
watch -n 1 rocm-smi --showmemuse --showuse
```

This helps distinguish between:

- Allocated GPU memory
- GPU compute utilization
- Model memory remaining allocated while idle
- Additional activity during inference

High compute utilization during token generation does not necessarily indicate a problem.

VRAM allocation is a separate resource consideration because other desktop applications also require GPU memory.

## Image Generation Experiments

The project originally included experiments with local image generation.

The image generation stack used:

```text
Ubuntu
   |
   v
AMDGPU / ROCm
   |
   v
PyTorch
   |
   v
ComfyUI
   |
   v
SDXL models
```

Models tested included:

- SDXL Lightning
- Juggernaut XL v9

The repository contains a saved ComfyUI workflow for comparing model outputs using controlled settings such as prompt, resolution, and seed.

The project subsequently shifted toward local LLM infrastructure, persistent application state, and RAG.

## Project Structure

```text
local-ai-lab/
├── .github/
│   └── workflows/
│       └── tests.yml
├── .vscode/
│   └── tasks.json
├── data/                         # Local generated data
│   ├── chat_history.db
│   ├── chroma/
│   ├── fastembed_cache/
│   └── rag_evaluations/
├── docs/
│   ├── architecture.md
│   ├── experiments.md
│   └── troubleshooting.md
├── evals/
│   └── frankenstein.json
├── knowledge/
│   ├── ai_notes.txt
│   ├── docker_notes.txt
│   ├── dune.txt
│   └── frankenstein.txt
├── scripts/
│   ├── monitor_gpu.sh
│   └── start_llama_server.sh
├── src/
│   ├── chunk_inspect.py
│   ├── config.py
│   ├── context_manager.py
│   ├── database.py
│   ├── evaluate_rag.py
│   ├── llm_client.py
│   ├── rag.py
│   ├── rag_demo.py
│   ├── retrieval.py
│   ├── retrieval_demo.py
│   └── vector_store.py
├── tests/
│   ├── test_context_manager.py
│   ├── test_database.py
│   ├── test_rag.py
│   ├── test_rag_evaluation.py
│   ├── test_retrieval.py
│   └── test_vector_store.py
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

The contents of `data/` are generated locally and are intentionally excluded from Git.

## Testing

The project uses pytest for automated Python testing.

### Native Testing

From the repository root:

```bash
pytest -q
```

For more detailed output:

```bash
python -m pytest -v
```

### Docker Testing

```bash
docker compose run --rm app python -m pytest -v
```

### Unit Test Coverage

Current tests cover:

- SQLite conversation creation and deletion
- Saving and loading messages
- Conversation pagination and ordering
- Context-budget calculations
- Retaining recent complete conversation turns
- Preserving the system message
- Rejection of requests that cannot fit
- Efficient history cutoff selection
- Document loading and chunking
- Vector-store indexing and synchronization
- Detection of added, changed, and deleted documents
- Retrieval behavior
- RAG prompt construction
- Evaluation phrase matching
- Evidence-rank selection
- Answer-checking rules

The latest local test run completed with 26 passing tests.

Most unit tests use controlled inputs, temporary data, or test doubles rather than making live inference requests.

Unit test success does not establish model-answer accuracy.

## Continuous Integration

GitHub Actions runs the Python test suite on pushes and pull requests.

The workflow:

- Uses Ubuntu 24.04
- Uses Python 3.12
- Installs dependencies from `requirements.txt`
- Checks Python syntax
- Executes pytest
- Uses read-only repository permissions
- Has a five-minute job timeout

CI validates the Python application and test logic.

It does not run local Qwen inference or full FastEmbed/Chroma retrieval evaluations against the development knowledge base.

Those integration checks require the appropriate local runtime, models, and persistent data.

## Privacy and Local Data

The project is designed around local inference and locally stored information.

The llama.cpp server binds to:

```text
127.0.0.1
```

rather than being exposed to the local network.

No cloud model API is required for the current inference workflow.

Initial installation of Python packages or embedding-model files may require downloading dependencies.

Local application data is excluded from Git, including:

- SQLite conversation history
- Chroma databases
- Chroma backups
- FastEmbed model caches
- Generated evaluation reports
- Environment configuration files

The example evaluation questions and public-domain test document remain part of the source repository.

## Current Limitations

The project is a working development lab, not a production-ready AI service.

Current limitations include:

- The persistent chat client and document RAG demo are separate workflows.
- Conversation context selection is primarily recency-based.
- Retrieval quality varies with document content, chunking, and question wording.
- Relevant evidence may appear near the bottom of the retrieval limit.
- The chunking strategy uses word counts rather than model-token counts.
- Changing chunking or embedding settings may require manually rebuilding the index.
- Generated answers can include unsupported statements even when relevant evidence was retrieved.
- Automated answer checks use phrase-based heuristics.
- Generated answers do not yet provide verified passage-level citations.
- Docker networking is currently optimized for the Ubuntu development environment.
- Hardware-specific inference configuration may not work unchanged on other machines.

## Planned Improvements

Potential next steps include:

- Integrating document retrieval into persistent conversations
- Improving retrieval relevance and ranking
- Source-linked citations and more precise provenance
- Better handling of document and model version changes
- Additional document formats
- Improved long-term conversation retrieval
- Private or non-persistent sessions
- Tool calling and external integrations
- Additional logging and observability
- More portable runtime configuration
- Additional application interfaces

## Project Purpose

Local AI Lab is an engineering exercise in building and understanding the surrounding systems required to operate local AI applications.

The work includes:

- Linux processes and permissions
- AMD GPU compute
- Model loading and inference
- HTTP APIs
- Streaming responses
- Python application architecture
- SQL and relational persistence
- Context-window management
- Algorithmic efficiency
- Embeddings and vector search
- Incremental data processing
- Retrieval-Augmented Generation
- Automated testing
- Evaluation and regression testing
- Docker and container networking
- GitHub Actions
- Debugging across multiple infrastructure layers

The emphasis is on reproducibility, local control, and understanding how these components interact.
