# Warehouse Management RAG Service

Modular, high-performance Retrieval-Augmented Generation (RAG) backend powered by **FastAPI**, **Qdrant**, **MongoDB**, **Redis**, and **Google Gemini**.

## Repository Structure

```
wm_rag_service/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI routes for RAG search and assistant response
│   ├── models.py            # Pydantic request/response schemas
│   ├── database.py          # Qdrant, Redis, and MongoDB connection manager
│   ├── processor.py         # Document chunking and text preprocessing
│   ├── embeddings.py        # Embedding generation logic
│   ├── retriever.py         # Vector search from Qdrant
│   ├── reranker.py          # Reranking retrieved results
│   ├── rag_pipeline.py      # Complete RAG flow: retrieve → rerank → generate
│   └── llm_client.py        # Gemini/LangChain LLM connection
│
├── scripts/
│   ├── __init__.py
│   ├── ingest_documents.py  # Ingest warehouse docs into Qdrant
│   ├── refresh_index.py     # Rebuild or update vector index
│   └── test_query.py        # Test RAG question-answer flow
│
├── data/
│   ├── raw/                 # Uploaded warehouse manuals, SOPs, BRDs
│   └── processed/           # Cleaned chunks or metadata files
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- Running instances of:
  - **Qdrant Vector Database** (port 6333)
  - **MongoDB** (port 27017)
  - **Redis** (port 6379)

### Installation

1. Clone the repository and navigate to the project directory:
   ```bash
   cd wm_rag
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables by copying `.env.example` to `.env` and updating the values:
   ```bash
   cp .env.example .env
   ```

### Execution & Scripts

#### 1. Reset / Refresh Vector Index
Create or clear the Qdrant collection:
```bash
python scripts/refresh_index.py
```

#### 2. Ingest Documents
Index manuals (SOPs, PDF files, etc.) from local storage or S3:
```bash
# Ingest local file
python scripts/ingest_documents.py data/raw/manual.pdf "Vehicle Model"

# Ingest from S3 key
python scripts/ingest_documents.py "s3-key-to-manual.pdf" "Vehicle Model"
```

#### 3. Run a Query Test CLI
Test RAG flow locally without launching the API server:
```bash
python scripts/test_query.py "How do I jumpstart the EV battery?" "Tata Nexon EV"
```

#### 4. Run the API Server
Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```
The interactive docs will be available at `http://localhost:8000/docs`.
