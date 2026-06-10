# Warehouse RAG — Intelligent Warehouse Management System

A specialized Retrieval-Augmented Generation (RAG) pipeline and API service built for Warehouse Management Systems (WMS). Powered by Google Gemini, Qdrant vector search, and custom warehouse-domain reranking.

---

## Table of Contents

1. [What is This?](#what-is-this)
2. [Architecture Overview](#architecture-overview)
3. [Quick Start](#quick-start)
4. [API Endpoints](#api-endpoints)
5. [Configuration Guide](#configuration-guide)
6. [How RAG Works](#how-rag-works)
7. [File Structure](#file-structure)
8. [Scripts](#scripts)
9. [Troubleshooting](#troubleshooting)

---

## What is This?

`wm_rag` is a **production-ready RAG pipeline** designed specifically for warehouse operations:

1. You feed it warehouse documents (manuals, SOPs, audit trails, OCR scans).
2. Documents are chunked with warehouse-aware segmentation (recognizing sections, warnings, procedures).
3. When a user asks a question, it **retrieves** the most relevant document chunks via Qdrant vector search.
4. Chunks are **reranked** using a custom warehouse-domain scorer (severity, section matching, word overlap).
5. The query + context is sent to **Google Gemini** for an actionable, JSON-structured response.

---

## Architecture Overview

```
┌──────────────────────────────────────────────┐
│            External App / Spring Boot         │
│           (sends warehouse queries)           │
└────────────────────┬─────────────────────────┘
                     │  POST /api/ai/analyze
                     ▼
┌──────────────────────────────────────────────┐
│          FastAPI Service (app/main.py)        │
│  ┌────────────────────────────────────────┐  │
│  │       RAGPipeline (app/rag_pipeline.py) │  │
│  │  ┌──────────┐ ┌─────────┐ ┌─────────┐ │  │
│  │  │Embedder  │ │Reranker │ │Generator│ │  │
│  │  │(MPNET)   │ │(Custom) │ │(Gemini) │ │  │
│  │  └────┬─────┘ └────┬────┘ └────┬────┘ │  │
│  └───────┼────────────┼───────────┼──────┘  │
│          ▼            ▼           ▼          │
│  ┌──────────┐ ┌─────────────┐ ┌──────────┐  │
│  │  Qdrant  │ │   Redis     │ │ MongoDB  │  │
│  │ (Vector) │ │  (Cache)    │ │ (Audit)  │  │
│  └──────────┘ └─────────────┘ └──────────┘  │
└──────────────────────────────────────────────┘
```

---

## Quick Start

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Set environment variables
Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```

Required variables:
```
GEMINI_API_KEY=your_gemini_api_key
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
```

### Step 3: Launch the service
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Step 4: Test with a query
```bash
curl -X POST "http://localhost:8000/api/ai/analyze" \
     -H "Content-Type: application/json" \
     -d '{
           "ocrDocumentId": "DOC-001",
           "title": "Inventory Check",
           "description": "Where is SKU123 located?",
           "documentType": "StockReport",
           "warehouseId": "WH001",
           "userId": 101,
           "attemptCount": 1
         }'
```

---

## API Endpoints

### 1. `POST /api/ai/analyze`
Analyze warehouse documents with RAG-powered AI.

**Request:**
```json
{
  "ocrDocumentId": "DOC-001",
  "title": "Inventory Check",
  "description": "Where is SKU123 located in Zone A?",
  "documentType": "StockReport",
  "warehouseId": "WH001",
  "userId": 101,
  "attemptCount": 1,
  "auditTrail": [],
  "followUpQuestion": null,
  "previousResponse": null
}
```

**Response:**
```json
{
  "suggestion": "{\"analysis_summary\": \"...\", \"recommended_actions\": [...]}",
  "confidence": 0.85,
  "predictedCategory": "Inventory Management",
  "status": "SUCCESS"
}
```

### 2. `POST /api/rag/ingest`
Ingest OCR document text into the vector store.

**Request:**
```json
{
  "ocr_document_id": "DOC-001",
  "document_type": "StockReport",
  "warehouse_id": "WH001",
  "text": "Product SKU123 is stored in Zone A Rack R2 Bin B5..."
}
```

### 3. `GET /status`
Health check endpoint.

---

## Configuration Guide

### `configs/rag_config.yaml`
Default parameters for the RAG pipeline:
```yaml
embedder:
  provider: "sentence-transformers"
  model: "all-mpnet-base-v2"   # 768-dim embeddings

vector_store:
  provider: "qdrant"
  index_name: "warehouse-index"

generator:
  provider: "gemini"
  model: "gemini-2.5-flash"

chunker:
  provider: "warehouse"
  chunk_size: 500
  chunk_overlap: 50

retrieval:
  top_k: 8
  rerank: true
  rerank_top_k: 4
```

---

## How RAG Works

When a warehouse operator submits a query:

1. **Embed Query**: The query is converted to a 768-dim vector using `all-mpnet-base-v2`.
2. **Vector Search**: Qdrant retrieves the top-8 most similar document chunks.
3. **Rerank**: The custom `WarehouseReranker` rescores results using:
   - Semantic similarity (60%)
   - Word overlap (20%)
   - Section alignment (10%)
   - Severity boost (10%)
4. **Generate**: Top-4 reranked chunks + the query are sent to Gemini for synthesis.
5. **Cache & Persist**: The response is cached in Redis and logged to MongoDB.

---

## File Structure

```
wm_rag/
├── app/                          # FastAPI application layer
│   ├── main.py                   # API endpoints and lifespan
│   ├── models.py                 # Pydantic request/response models
│   ├── rag_pipeline.py           # WMS-specific RAG orchestrator
│   ├── database.py               # Redis, Qdrant, MongoDB clients
│   └── s3_handler.py             # Optional S3 file operations
│
├── rag_core/                     # Core reusable RAG library
│   ├── __init__.py               # Public API exports
│   ├── config.py                 # YAML/dict config parser
│   ├── types.py                  # Core data types (Chunk, SearchResult)
│   ├── pipeline.py               # Generic RAG pipeline orchestrator
│   ├── embedders/                # Text embedding providers
│   │   ├── base.py               # BaseEmbedder interface
│   │   └── sentence_transformer.py
│   ├── vector_stores/            # Vector database providers
│   │   ├── base.py               # BaseVectorStore interface
│   │   └── qdrant_store.py
│   ├── generators/               # LLM generation providers
│   │   ├── base.py               # BaseGenerator interface
│   │   └── gemini.py
│   ├── chunkers/                 # Document chunking providers
│   │   ├── base.py               # BaseChunker interface
│   │   └── warehouse.py          # WMS-specific chunker
│   └── rerankers/                # Result reranking providers
│       ├── base.py               # BaseReranker interface
│       └── warehouse.py          # WMS-specific reranker
│
├── configs/
│   ├── rag_config.yaml           # Default pipeline configuration
│   └── router_config.yaml        # Router settings
│
├── scripts/
│   ├── ingest_documents.py       # Bulk document ingestion
│   ├── recreate_collection.py    # Reset Qdrant collection
│   ├── refresh_index.py          # Refresh vector index
│   ├── run_ingest_test.py        # Quick ingest test
│   └── test_query.py             # Standalone query test
│
├── tests/
│   └── test_pipeline.py          # Pipeline unit/integration tests
│
├── pyproject.toml                # Python package configuration
├── requirements.txt              # Direct dependency requirements
├── .env.example                  # Environment variable template
└── .gitignore                    # Git ignore patterns
```

---

## Scripts

### Ingest documents
```bash
python scripts/ingest_documents.py <file_path> <document_type> [clear_existing] [warehouse_id]
```

### Recreate Qdrant collection
```bash
python scripts/recreate_collection.py
```

### Test a query
```bash
python scripts/test_query.py "Where is SKU123?" StockReport
```

---

## Troubleshooting

### "Gemini API key not found"
Ensure `GEMINI_API_KEY` is set in your `.env` file. The system will run in mock mode if the key is missing.

### "Qdrant connection timeout"
- Verify `QDRANT_URL` and `QDRANT_API_KEY` in `.env`
- The default timeout is 30 seconds for cloud instances
- Run `python scripts/recreate_collection.py` to ensure the collection exists

### "Redis/MongoDB connection errors"
These are optional for local development. The service will continue running with warnings if Redis or MongoDB are unavailable. Caching and audit logging will be degraded.

### "No results returned"
- Ensure documents have been ingested: `POST /api/rag/ingest`
- Check that the Qdrant collection exists: `python scripts/recreate_collection.py`
- Verify the embedding model matches (768-dim for `all-mpnet-base-v2`)
