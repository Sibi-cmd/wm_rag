# rag_core — Plug-and-Play RAG Library & API Service

A standalone, reusable, multi-tenant Retrieval-Augmented Generation (RAG) library and API service. Configure once, run anywhere.

No domain-specific logic. No hardcoded providers. Strict data isolation. Pure plug-and-play.

---

## Table of Contents

1. [What is This?](#what-is-this)
2. [Architecture Overview](#architecture-overview)
3. [Quick Start (5 Minutes)](#quick-start)
4. [Standalone API Middleman Service](#standalone-api-middleman-service)
5. [Advanced Multi-Database RAG Router](#advanced-multi-database-rag-router)
6. [Configuration Guide](#configuration-guide)
7. [Step-by-Step: How RAG Works](#step-by-step-how-rag-works)
8. [Usage Guide](#usage-guide)
9. [Supported Providers](#supported-providers)
10. [File Structure](#file-structure)
11. [Troubleshooting](#troubleshooting)

---

## What is This?

`rag_core` is a **reusable Python library and microservice** that provides any project with a complete, production-ready RAG pipeline.

**RAG** (Retrieval-Augmented Generation) means:
1. You feed it your documents or databases.
2. When a user asks a question, it **finds** the most relevant pieces of information.
3. It sends those pieces + the question to an LLM (like OpenAI GPT-4o).
4. The LLM provides an **accurate, hallucination-free answer based on your actual data**.

**Multi-Tenant & Dynamic Routing** means:
- You can run the RAG system as a standalone **Middleman API Service** (using FastAPI).
- External web applications (like Django, Flask, or FastAPI) can query the RAG service with a database URL (`db_url`), database name (`db_name`), and query.
- The RAG Middleman dynamically connects to the target database, retrieves/indexes the relevant text records, and returns the RAG response with **100% data separation** (Database A never accesses or leaks Database B's data).

---

## Architecture Overview

```
             ┌────────────────────────────────────────────────────────┐
             │                  External App (Client)                 │
             │                    (Django / FastAPI)                  │
             └───────────────────────────┬────────────────────────────┘
                                         │
                   POST /ask             │
                   {db_url, db_name,     │
                    query, mapping}      ▼
             ┌────────────────────────────────────────────────────────┐
             │                     RAG Middleman                      │
             │                    (FastAPI Server)                    │
             │                                                        │
             │   ┌────────────────────────────────────────────────┐   │
             │   │                   RAGRouter                    │   │
             │   │  - LRU Connection & Pipeline Cache             │   │
             │   │  - Dynamic SQL Data Fetcher & Chunker          │   │
             │   │  - Persistent Disk Vector Cache                │   │
             │   └───────────────────────┬────────────────────────┘   │
             └───────────────────────────┼────────────────────────────┘
                                         │  Routes query ONLY
                                         │  to respective DB
                                         ▼
             ┌────────────────────────────────────────────────────────┐
             │                   Isolated Databases                   │
             │                                                        │
             │   ┌───────────────┐  ┌───────────────┐  ┌───────────┐  │
             │   │  Inspections  │  │  Diagnostics  │  │ Fleet DB  │  │
             │   │    (SQL/V)    │  │    (SQL/V)    │  │  (SQL/V)  │  │
             │   └───────────────┘  └───────────────┘  └───────────┘  │
             └────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Step 1: Set environment variables
In your `.env` file or environment:
```bash
OPENAI_API_KEY=your_openai_api_key
```

### Step 2: Launch the Standalone API Middleman
Spin up the FastAPI server locally:
```bash
python examples/api_service.py
```
* **Dashboard URL**: [http://localhost:8000/](http://localhost:8000/) — Beautiful dark-theme glassmorphism monitoring panel showing active cached pools.
* **Interactive OpenAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs) — Test queries directly from the browser!

### Step 3: Query Your RAG Middleman
To query the service from your Django/FastAPI app, send a POST request to `/ask`:
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
           "db_name": "inspections_alias",
           "query": "What are the notes on the Bridge cracks?"
         }'
```

---

## Standalone API Middleman Service

When deployed, `rag_core` acts as a middleman, handling the heavy embedding and LLM workloads so your main web services stay lightweight.

### 🌐 Endpoints

#### 1. `POST /ask`
Submit a question to be routed dynamically to a specific database.
* **Request Payload**:
  ```json
  {
    "db_url": "sqlite:///C:/path/to/database.sqlite",
    "db_name": "client_inspections",
    "query": "Bridge cracks in concrete",
    "top_k": 3,
    "filters": {"inspector_id": "John"},
    "schema_mapping": {
      "table_name": "field_reports",
      "text_columns": ["notes", "summary"],
      "metadata_columns": ["inspector_id", "date"]
    },
    "force_sync": false
  }
  ```
* **Response**:
  ```json
  {
    "answer": "John Doe noticed visible concrete cracks in the Bridge A abutment...",
    "sources": ["field_reports"],
    "confidence": 0.89,
    "chunks_used": 2,
    "db_name": "client_inspections"
  }
  ```

#### 2. `POST /sync`
Refresh or re-index the RAG cache manually for a target database when underlying source data changes.
* **Request Payload**:
  ```json
  {
    "db_url": "sqlite:///C:/path/to/database.sqlite",
    "db_name": "client_inspections",
    "schema_mapping": {
      "table_name": "field_reports",
      "text_columns": ["notes"],
      "metadata_columns": ["inspector_id"]
    }
  }
  ```

#### 3. `GET /health`
Returns connection status, loaded model names, and base configurations.

#### 4. `GET /stats`
Retrieves connection pool statistics and active memory cache metrics.

---

## Advanced Multi-Database RAG Router

The **`RAGRouter`** module is the brain of the multi-database architecture, implementing high-grade engineering practices:

* **⚡ Persistent Index Caching (No Cold-Starts)**: Connecting to databases and embedding files on-the-fly is slow. `RAGRouter` builds a vector index the first time it is queried and **saves the index to disk** (under `./vector_dbs/{db_name}/`). Subsequent queries load from disk in **sub-milliseconds**.
* **🔒 Secure Connection Aliases**: To avoid sending database credentials over HTTP requests, you register database connections in `configs/router_config.yaml` under aliases (like `inspections_alias`). Your client web apps just send `db_name: "inspections_alias"` and leave `db_url` empty.
* **🧠 LRU Cache (RAM Protection)**: Keeps active RAGPipelines and databases loaded in memory. If a pipeline is inactive for more than 30 minutes, or if the cache limit is exceeded, it cleanly unloads it from RAM.
* **📋 Dynamic Schema Mapping**: Relational databases have different structures. You can specify different table names, text columns to embed, and columns to preserve as metadata dynamically in each request.

---

## Configuration Guide

The dynamic service is controlled by two YAML configuration files located in the `configs/` directory.

### 1. `configs/router_config.yaml`
Defines server limits, cache settings, and database connection aliases:
```yaml
# Base pipeline settings path
base_rag_config: "configs/rag_config.yaml"

# LRU Cache settings
router:
  max_cached_pipelines: 50     # Maximum active pipelines in RAM
  ttl_seconds: 1800            # Unload pipeline after 30 mins of inactivity
  local_index_base_dir: "./vector_dbs"

# Registry (Aliases) - Mask credentials
connections:
  inspections_alias: "sqlite:///C:/path/to/inspections.sqlite"
  fleet_alias: "sqlite:///C:/path/to/fleet.sqlite"
```

### 2. `configs/rag_config.yaml`
Defines default parameters for chunkers, generators, and embedders:
```yaml
embedder:
  provider: "sentence-transformers"
  model: "all-mpnet-base-v2"

vector_store:
  provider: "faiss"
  index_path: "./faiss_index"
  dimensions: 768

generator:
  provider: "openai"
  model: "gpt-4o-mini"
  max_retries: 3
  timeout: 30
```

---

## Step-by-Step: How RAG Works

When a client queries the middleman via `POST /ask`:
1. **Route Connection**: `RAGRouter` extracts `db_url` / `db_name` and checks if it's already cached. If not, it establishes a dynamic connection.
2. **Build Index (if new)**: If the index file isn't found on disk, it queries the target table, splits records into chunks, embeds them, and saves the FAISS index to disk.
3. **Retrieve Chunks**: It converts the user query into a vector, queries the isolated FAISS database, and filters/reranks candidate chunks.
4. **Generate Answer**: Renders the context into the prompt template and sends it to the LLM.
5. **Return Response**: Returns answer text, unique sources, and confidence scores.

---

## Usage Guide (Local Library Mode)

If you prefer to import `RAGRouter` directly in Python instead of running the web server:

```python
from rag_core import RAGRouter

# Initialize Router
router = RAGRouter(config_path="configs/router_config.yaml")

# Run query on the Bridge SQLite Database
response = router.ask_tenant(
    db_url="sqlite:///bridge_reports.sqlite",
    db_name="bridge_db",
    query="Are there concrete cracks reported?",
    schema_mapping={
        "table_name": "reports",
        "text_columns": ["notes"],
        "metadata_columns": ["inspector", "site"]
    }
)

print(response.text)
```

---

## File Structure

```
field_inspections/
├── rag_core/                         # Core reusable library (now at root!)
│   ├── __init__.py                   # Public API exports (includes RAGRouter)
│   ├── config.py                     # YAML config parser
│   ├── types.py                      # Standard datatypes (Chunk, RAGResponse)
│   ├── pipeline.py                   # Single RAGPipeline orchestrator
│   ├── router.py                     # Dynamic Multi-DB Router (LRU, Aliases, Sync)
│   ├── api.py                        # Standalone FastAPI service & Dashboard
│   ├── embedders/                    # Text embedding providers
│   ├── vector_stores/                # FAISS and Pinecone storage
│   ├── generators/                   # OpenAI & LLM interfaces
│   ├── chunkers/                     # PDF & MD document splitters
│   └── rerankers/                    # Hybrid relevance rankers
│
├── configs/
│   ├── rag_config.yaml               # Default pipeline configurations
│   └── router_config.yaml            # Connections alias and Router settings
│
├── tests/
│   ├── test_pipeline.py              # Pipeline unit/integration tests
│   └── test_router.py                # Multi-tenant isolation and cache tests
│
├── examples/
│   └── api_service.py                # Standalone API server launcher
│
├── pyproject.toml                    # Standard Python package specifications
├── requirements.txt                  # Direct dependency requirements
├── .env                              # Environment credentials (API keys)
└── .gitignore                        # Git ignore patterns (ignores vector_dbs/)
```

---

## Troubleshooting

### "SQL connection failed"
Ensure the connection string matches SQLAlchemy standards (e.g. `sqlite:///C:/path/to/db.sqlite` or `postgresql://user:pass@host/db`).

### "Cold-start delay on first query"
The first query on a new database indexes the tables. Subsequent queries load the cached files instantly. To prevent delay during client requests, trigger a background `/sync` request right after databases are updated.

### "Out of memory"
If serving hundreds of concurrent databases, adjust `max_cached_pipelines` and `ttl_seconds` in `configs/router_config.yaml` to release pipeline handles from RAM faster.
