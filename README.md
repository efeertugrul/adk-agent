# ADK Documentation RAG Agent

A Retrieval-Augmented Generation (RAG) assistant designed for the `google/adk-python` repository. This project crawls documentation, generates vector embeddings on CPU via FastEmbed, and exposes an agentic tool interface via `google-adk` and `gemini-3.5-flash-lite`.

It features both an interactive Web UI and an asynchronous streaming Command-Line Interface (CLI).

---

## Key Features

* **Decoupled System Architecture:** Clean separation between document ingestion (`ingest.py`), agent reasoning/tool retrieval (`agent.py`), and runner interfaces (`cli.py`).
* **Rate-Limit Aware Web Crawler:** Ingests raw GitHub documentation using `httpx`, tracking rate limits with automated backoff/countdown handling.
* **Local CPU Embeddings:** Uses FastEmbed (`BAAI/bge-small-en-v1.5`) to generate dense vector embeddings locally without external API costs or latency overhead.
* **Local Vector Store:** Lightweight, payload-safe vector storage powered by an embedded Qdrant instance.
* **Strict Tool Grounding:** Uses `google-adk`'s `FunctionTool` abstraction to provide Gemini 3.6 Flash with explicit context retrieval mechanisms.
* **Dual Interface Support:** Support for `adk web` developer tool UI and a streaming terminal CLI built with `InMemoryRunner`.

---

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Language & Environment** | Python 3.12, `uv` package manager |
| **Agent Framework** | `google-adk` |
| **LLM Engine** | `gemini-3.5-flash-lite` |
| **Vector Database** | Qdrant (Embedded / Local File Path) |
| **Embedding Model** | FastEmbed (`BAAI/bge-small-en-v1.5`) |

---

## Project Structure

```text
.
├── src/
│   └── adk_agent/
│       ├── __init__.py
│       ├── agent.py          # LlmAgent and Qdrant FunctionTool definition
│       ├── cli.py            # Async streaming CLI runner interface
│       └── ingest.py         # Rate-limit aware GitHub doc crawler & vector indexer
├── qdrant_db/                # Embedded local vector database storage
├── .env                      # API keys and configuration
├── pyproject.toml            # Dependencies and script entry points
└── README.md


## Setup & Configuration

### 1. Prerequisites & Installation

Make sure you have [uv](https://github.com/astral-sh/uv) installed (Python 3.12+):

```bash
# Install uv package manager
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# Clone the repository
git clone [https://github.com/efeertugrul/adk-agent.git](https://github.com/efeertugrul/adk-agent.git)
cd adk-agent

# Install all dependencies into virtual environment
uv sync
