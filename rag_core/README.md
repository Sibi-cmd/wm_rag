# rag_core — Plug-and-Play RAG Library

A standalone, reusable Retrieval-Augmented Generation (RAG) library. Configure once, use anywhere.
No domain-specific logic. No hardcoded providers. Pure plug-and-play.

---

## Table of Contents

1. [What is This?](#what-is-this)
2. [Architecture Overview](#architecture-overview)
3. [Quick Start (5 Minutes)](#quick-start)
4. [Installation](#installation)
5. [Configuration (rag_config.yaml)](#configuration)
6. [Step-by-Step: How RAG Works](#step-by-step-how-rag-works)
7. [Usage Guide](#usage-guide)
   - [Ingesting Documents](#1-ingesting-documents)
   - [Asking Questions](#2-asking-questions)
   - [Retrieving Without Generating](#3-retrieving-without-generating)
   - [Generating Without Retrieving](#4-generating-without-retrieving)
8. [Supported Providers](#supported-providers)
9. [Adding a New Provider](#adding-a-new-provider)
10. [Real-World Use Cases](#real-world-use-cases)
11. [File Structure](#file-structure)
12. [Troubleshooting](#troubleshooting)

---

## What is This?

`rag_core` is a **reusable Python library** that gives any project a fully working RAG pipeline.

**RAG** = Retrieval-Augmented Generation. It means:
1. You feed it your documents (PDFs, text files, markdown, etc.)
2. When a user asks a question, it **finds** the most relevant pieces of your documents
3. It sends those pieces + the question to an LLM (like OpenAI GPT-4o, DeepSeek, or Ollama)
4. The LLM gives an **accurate answer based on your actual data** — not hallucinations

**Plug-and-play** means:
- You don't write RAG logic yourself
- You configure a YAML file
- You call `pipeline.ask("your question")`
- It works for ANY domain — support bots, legal Q&A, medical FAQ, internal wikis, anything

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        RAGPipeline                          │
│                                                             │
│  ┌───────────┐   ┌──────────────┐   ┌──────────┐           │
│  │  Chunker   │   │  Embedder    │   │ Reranker │           │
│  │           │   │              │   │          │           │
│  │ Split docs │   │ Text→Vector  │   │ Re-score │           │
│  │ into pieces│   │              │   │ results  │           │
│  └─────┬─────┘   └──────┬───────┘   └────┬─────┘           │
│        │                │                 │                 │
│        ▼                ▼                 │                 │
│  ┌──────────────┐                         │                 │
│  │ Vector Store  │  ◄─────────────────────┘                 │
│  │              │                                           │
│  │ Store & find  │                                          │
│  │ similar docs  │                                          │
│  └──────┬───────┘                                           │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────┐                                            │
│  │  Generator   │                                           │
│  │             │                                            │
│  │ LLM call →  │                                            │
│  │ final answer│                                            │
│  └─────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
```

**Every box is swappable.** Change the YAML config → different provider loads. Your code stays the same.

---

## Quick Start

### Step 1: Set environment variables

```bash
# In your .env file or system environment:
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

### Step 2: Create a config file

```yaml
# rag_config.yaml
embedder:
  provider: "sentence-transformers"
  model: "all-mpnet-base-v2"

vector_store:
  provider: "pinecone"
  index_name: "my-project-index"
  api_key_env: "PINECONE_API_KEY"

generator:
  provider: "openai"
  model: "gpt-4o-mini"
  api_key_env: "OPENAI_API_KEY"
  max_retries: 3

chunker:
  chunk_size: 500
  chunk_overlap: 50

retrieval:
  top_k: 5
  rerank: true
  rerank_top_k: 3

prompt_template: |
  Use the following context to answer the question.
  If you cannot find the answer in the context, say "I don't have enough information."

  Context:
  {context}

  Question: {question}

  Answer:
```

### Step 3: Ingest your documents

```python
from rag_core import RAGPipeline

pipeline = RAGPipeline.from_config("rag_config.yaml")

# Feed it your documents (PDF, TXT, MD)
pipeline.ingest_files([
    "./docs/user_manual.pdf",
    "./docs/faq.md",
    "./docs/troubleshooting.txt",
])

print("Documents ingested!")
```

### Step 4: Ask questions

```python
answer = pipeline.ask("How do I reset my password?")

print(answer.text)        # "To reset your password, go to Settings > ..."
print(answer.sources)     # ["faq.md - Section: Account Management"]
print(answer.confidence)  # 0.89
```

**That's it.** Four steps. Works for any project.

---

## Installation

### Dependencies

```txt
# Core (required)
sentence-transformers
openai
python-dotenv
pyyaml
tiktoken

# Vector stores (install what you need)
pinecone          # for Pinecone cloud
faiss-cpu         # for local FAISS

# Document processing
langchain-text-splitters
langchain-community
pypdf
```

### Install

```bash
pip install -r requirements.txt
```

---

## Configuration

The YAML config file controls every component. Here's every option explained:

### Embedder

```yaml
embedder:
  provider: "sentence-transformers"   # Which embedding library to use
  model: "all-mpnet-base-v2"          # Which model (determines vector dimensions)
```

| Provider | Model Examples | Vector Dimensions | Speed | Quality |
|----------|---------------|-------------------|-------|---------|
| `sentence-transformers` | `all-mpnet-base-v2` | 768 | Medium | High |
| `sentence-transformers` | `all-MiniLM-L6-v2` | 384 | Fast | Medium |

### Vector Store

```yaml
# Option A: Pinecone (cloud — good for production)
vector_store:
  provider: "pinecone"
  index_name: "my-index"              # Your Pinecone index name
  api_key_env: "PINECONE_API_KEY"     # Environment variable holding the API key

# Option B: FAISS (local — good for development/testing)
vector_store:
  provider: "faiss"
  index_path: "./faiss_index"         # Local folder to save/load the index
  dimensions: 768                     # Must match your embedder's output dimensions
```

### Generator (LLM)

```yaml
generator:
  provider: "openai"                  # Which LLM to call
  model: "gpt-4o-mini"                # Which model variant (or "deepseek-chat")
  api_key_env: "OPENAI_API_KEY"       # Environment variable holding the API key
  max_retries: 3                      # How many times to retry on failure
  timeout: 30                         # Seconds to wait per request
```

| Provider | Model Examples | Best For |
|----------|---------------|----------|
| `openai` | `gpt-4o-mini` | Fast, highly cost-effective Q&A |
| `openai` | `gpt-4o` | Nuanced, advanced reasoning |
| `openai` | `deepseek-chat` | Incredible value and speed |

### Chunker

```yaml
chunker:
  chunk_size: 500                     # Max tokens per chunk
  chunk_overlap: 50                   # Overlap between chunks (prevents context loss)
```

**Guidelines:**
- `chunk_size: 300-500` — Good for precise Q&A (support bots, FAQ)
- `chunk_size: 500-1000` — Good for detailed explanations (manuals, legal docs)
- `chunk_overlap: 50-100` — Ensures important info at chunk boundaries isn't lost

### Retrieval

```yaml
retrieval:
  top_k: 5                           # How many chunks to retrieve from vector store
  rerank: true                       # Whether to re-score results for better accuracy
  rerank_top_k: 3                    # After reranking, keep only the best N
```

### Prompt Template

```yaml
prompt_template: |
  Use the following context to answer the question.
  If you cannot find the answer in the context, say "I don't know."

  Context:
  {context}

  Question: {question}

  Answer:
```

**Two required placeholders:**
- `{context}` — Gets replaced with the retrieved document chunks
- `{question}` — Gets replaced with the user's question

You can customize this for any tone, format, or domain:

```yaml
# Example: Strict JSON output
prompt_template: |
  You are an assistant. Answer based ONLY on the context below.
  Return your answer as JSON with keys: "answer", "sources", "confidence".

  Context: {context}
  Question: {question}

# Example: Friendly chatbot
prompt_template: |
  You are a friendly helper. Use the information below to assist the user.
  Be conversational and helpful. If you're unsure, say so honestly.

  Information: {context}
  User asked: {question}
```

---

## Step-by-Step: How RAG Works

Here's exactly what happens when you call `pipeline.ask("How do I reset my password?")`:

### Step 1: EMBED the question

```
Input:  "How do I reset my password?"
Output: [0.23, -0.41, 0.87, 0.12, ...]  (a vector of 768 numbers)
```

The Embedder converts your question into a numerical representation (vector) that captures its **meaning**. Similar questions produce similar vectors.

### Step 2: SEARCH the vector store

```
Input:  [0.23, -0.41, 0.87, 0.12, ...]  (the question vector)
Output: Top 5 most similar document chunks
```

The Vector Store compares your question vector against all stored document vectors and returns the most similar ones. This is how the system "finds" relevant information.

### Step 3: RERANK (optional)

```
Input:  5 candidate chunks
Output: 3 best chunks (re-scored using keyword overlap + similarity)
```

The Reranker applies additional scoring (keyword matching, metadata boosts) to pick the truly best matches from the candidates.

### Step 4: BUILD the prompt

```
Input:  Template + context chunks + question
Output: A complete prompt string ready for the LLM

  "Use the following context to answer the question.

   Context:
   [Chunk 1: To reset your password, navigate to Settings > Security > Reset Password...]
   [Chunk 2: If you forgot your password, click 'Forgot Password' on the login page...]

   Question: How do I reset my password?

   Answer:"
```

### Step 5: GENERATE the answer

```
Input:  The complete prompt
Output: "To reset your password, go to Settings > Security > Reset Password.
         Alternatively, click 'Forgot Password' on the login page and
         follow the email instructions."
```

The Generator sends the prompt to the selected LLM (OpenAI, DeepSeek, or Ollama) and returns the response.

### Step 6: RETURN structured result

```python
RAGResponse(
    text="To reset your password, go to Settings > Security > ...",
    sources=["faq.md - Section: Account Management", "manual.pdf - Page 12"],
    confidence=0.89,
    chunks_used=3
)
```

---

## Usage Guide

### 1. Ingesting Documents

Ingestion = feeding your documents into the system so it can search them later.

```python
from rag_core import RAGPipeline

pipeline = RAGPipeline.from_config("rag_config.yaml")

# Ingest from file paths
pipeline.ingest_files([
    "./docs/manual.pdf",
    "./docs/faq.md",
    "./docs/guide.txt",
])

# Ingest raw text directly
pipeline.ingest_texts([
    {"text": "Our return policy allows returns within 30 days.", "metadata": {"source": "policy", "section": "Returns"}},
    {"text": "Shipping takes 3-5 business days.", "metadata": {"source": "policy", "section": "Shipping"}},
])
```

**What happens during ingestion:**
1. Files are read and parsed (PDF → text, MD → text, etc.)
2. Text is split into chunks (using the Chunker)
3. Each chunk is converted to a vector (using the Embedder)
4. Vectors + metadata are stored in the Vector Store

**You only need to ingest once.** After that, the data stays in your vector store.

### 2. Asking Questions

```python
# Simple ask
answer = pipeline.ask("What is the return policy?")
print(answer.text)

# Ask with metadata filters (if your vector store supports it)
answer = pipeline.ask(
    "What is the return policy?",
    filters={"section": "Returns"}
)

# Ask with custom top_k (override config)
answer = pipeline.ask("What is the return policy?", top_k=10)
```

### 3. Retrieving Without Generating

Sometimes you just want the relevant chunks, without calling the LLM.

```python
results = pipeline.retrieve("battery issue")

for result in results:
    print(f"Score: {result.score}")
    print(f"Text: {result.text}")
    print(f"Source: {result.metadata.get('source')}")
    print("---")
```

Use this when:
- You want to build your own prompt
- You're debugging retrieval quality
- You want to display search results in a UI

### 4. Generating Without Retrieving

If you already have context and just want to call the LLM:

```python
answer = pipeline.generate(
    prompt="Given that the return policy is 30 days, can I return an item bought 25 days ago?",
)
print(answer)  # "Yes, since 25 days is within the 30-day return window..."
```

---

## Supported Providers

### Embedders

| Provider ID | Library | Models | Install |
|------------|---------|--------|---------|
| `sentence-transformers` | sentence-transformers | `all-mpnet-base-v2`, `all-MiniLM-L6-v2`, etc. | `pip install sentence-transformers` |

### Vector Stores

| Provider ID | Type | Best For | Install |
|------------|------|----------|---------|
| `pinecone` | Cloud | Production, large-scale | `pip install pinecone` |
| `faiss` | Local | Development, testing, offline | `pip install faiss-cpu` |

### Generators (LLMs)

| Provider ID | API | Models | Install |
|------------|-----|--------|---------|
| `openai` | OpenAI-compatible APIs (OpenAI, Ollama, DeepSeek) | `gpt-4o-mini`, `deepseek-chat`, custom models | `pip install openai` |

---

## Adding a New Provider

Every component follows the same pattern. Here's how to add a new one:

### Example: Adding an OpenAI Generator

**Step 1:** Create `rag_core/generators/openai.py`

```python
import os
from .base import BaseGenerator


class OpenAIGenerator(BaseGenerator):
    """Generator that uses OpenAI's API."""

    def __init__(self, config: dict):
        self.model = config.get("model", "gpt-4o-mini")
        api_key = os.getenv(config.get("api_key_env", "OPENAI_API_KEY"))
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.choices[0].message.content
```

**Step 2:** Register it in `rag_core/generators/__init__.py`

```python
GENERATOR_REGISTRY = {
    "openai": "rag_core.generators.openai.OpenAIGenerator",
    "anthropic": "rag_core.generators.anthropic.AnthropicGenerator",   # ← add this line
}
```

**Step 3:** Use it in your config

```yaml
generator:
  provider: "openai"
  model: "gpt-4o-mini"
  api_key_env: "OPENAI_API_KEY"
```

**That's it.** Same pattern works for adding a new Embedder, Vector Store, Chunker, or Reranker.

### The Base Class Pattern

Every component extends a base class with a simple interface:

```python
# BaseEmbedder
class BaseEmbedder:
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

# BaseVectorStore
class BaseVectorStore:
    def upsert(self, vectors: list[dict]) -> None: ...
    def search(self, vector: list[float], top_k: int, filters: dict = None) -> list[SearchResult]: ...
    def delete(self, ids: list[str] = None, delete_all: bool = False) -> None: ...

# BaseGenerator
class BaseGenerator:
    def generate(self, prompt: str, **kwargs) -> str: ...

# BaseChunker
class BaseChunker:
    def chunk(self, text: str, metadata: dict = None) -> list[Chunk]: ...
    def chunk_file(self, file_path: str) -> list[Chunk]: ...

# BaseReranker
class BaseReranker:
    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]: ...
```

Implement these methods → your provider works with the pipeline automatically.

---

## Real-World Use Cases

### Use Case 1: EV Fleet Diagnostics

```yaml
# ev_config.yaml
embedder:
  provider: "sentence-transformers"
  model: "all-mpnet-base-v2"
vector_store:
  provider: "pinecone"
  index_name: "ev-manual-index"
  api_key_env: "PINECONE_API_KEY"
generator:
  provider: "openai"
  model: "gpt-4o-mini"
  api_key_env: "OPENAI_API_KEY"
chunker:
  chunk_size: 500
  chunk_overlap: 50
retrieval:
  top_k: 8
  rerank: true
  rerank_top_k: 4
prompt_template: |
  You are an EV diagnostic assistant. Use ONLY the manual context below.
  Keep answers practical and easy for a non-mechanic driver to understand.

  Manual Context:
  {context}

  Driver's Question: {question}

  Provide a clear, actionable answer:
```

```python
pipeline = RAGPipeline.from_config("ev_config.yaml")
pipeline.ingest_files(["./manuals/nexon_ev.pdf", "./manuals/mg_zs_ev.pdf"])
answer = pipeline.ask("Battery draining fast despite full charge")
```

### Use Case 2: Customer Support Bot

```yaml
# support_config.yaml
embedder:
  provider: "sentence-transformers"
  model: "all-MiniLM-L6-v2"          # faster, smaller model
vector_store:
  provider: "faiss"                   # local, no cloud cost
  index_path: "./support_index"
  dimensions: 384
generator:
  provider: "openai"
  model: "gpt-4o-mini"
  api_key_env: "OPENAI_API_KEY"
chunker:
  chunk_size: 300                     # smaller chunks for precise FAQ answers
  chunk_overlap: 30
retrieval:
  top_k: 3
  rerank: false                       # FAQs are usually precise enough
prompt_template: |
  You are a friendly customer support assistant.
  Answer the customer's question using ONLY the information below.
  Be concise and helpful. If unsure, suggest contacting support@company.com.

  Knowledge Base:
  {context}

  Customer: {question}

  Assistant:
```

### Use Case 3: Legal Document Analysis

```yaml
# legal_config.yaml
embedder:
  provider: "sentence-transformers"
  model: "all-mpnet-base-v2"
vector_store:
  provider: "pinecone"
  index_name: "legal-docs"
  api_key_env: "PINECONE_API_KEY"
generator:
  provider: "openai"
  model: "gpt-4o"                     # highly advanced model for legal text
  api_key_env: "OPENAI_API_KEY"
chunker:
  chunk_size: 800                     # larger chunks to preserve legal context
  chunk_overlap: 100
retrieval:
  top_k: 10
  rerank: true
  rerank_top_k: 5
prompt_template: |
  You are a legal research assistant. Answer based STRICTLY on the provided documents.
  Cite specific clauses or sections. Do NOT provide legal advice — only summarize what the documents say.

  Documents:
  {context}

  Question: {question}

  Summary:
```

---

## File Structure

```
field_inspections/
├── src/
│   └── rag_core/                     # Core reusable library
│       ├── __init__.py               # Public API exports
│       ├── config.py                 # YAML config parser
│       ├── types.py                  # Standard datatypes (Document, RAGResponse)
│       ├── pipeline.py               # Main RAGPipeline orchestrator
│       ├── ingest.py                 # Batch ingestion helper
│       ├── embedders/                # Text embedding providers
│       ├── vector_stores/            # FAISS and Pinecone storage
│       ├── generators/               # OpenAI & Ollama LLM interfaces
│       ├── chunkers/                 # PDF & MD document splitters
│       └── rerankers/                # Hybrid relevance rankers
│
├── configs/
│   └── rag_config.yaml               # Default configuration settings
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py              # Automated test suite
│
├── examples/
│   ├── simple_qa.py                  # Interactive text Q&A example
│   └── ingest_directory.py           # Directory ingestion example
│
├── pyproject.toml                    # Standard Python package specifications
├── requirements.txt                  # Direct dependency requirements
├── .env.example                      # Template for secrets
└── .env                              # Active credentials and settings
```

---

## Troubleshooting

### "OPENAI_API_KEY not set"
→ Add `OPENAI_API_KEY=your_key` to your `.env` file or environment variables if calling standard OpenAI cloud models. If using Ollama, ensure your local server is running.

### "PINECONE_API_KEY not set"
→ Add `PINECONE_API_KEY=your_key` to your `.env` file. Or switch the `vector_store` provider to `faiss` in your YAML for fully offline local execution.

### "Index not found" (Pinecone)
→ Create the index in the Pinecone dashboard first. Make sure `index_name` in your YAML configuration matches exactly.

### "Dimension mismatch"
→ Your embedder's output dimensions must match your vector store's index dimensions.
  - `all-mpnet-base-v2` → 768 dimensions
  - `all-MiniLM-L6-v2` → 384 dimensions

### "No results returned"
→ Make sure you've ingested documents first (`pipeline.ingest_files([...])`).
→ Check that your query is related to the ingested content.

---

## Summary

| Step | What You Do | What rag_core Does |
|------|------------|-------------------|
| **1. Configure** | Adjust configs/rag_config.yaml | Dynamically instantiates the chosen providers |
| **2. Ingest** | Call `pipeline.ingest_files([...])` | Parses files → chunks → embeds → stores vectors |
| **3. Ask** | Call `pipeline.ask("your question")` | Embeds query → searches → reranks → calls LLM |
| **4. Get answer** | Use the response object | Returns answer text, unique sources, and confidence |
| **5. Next project?** | Write new YAML + supply new files | Reuses the same package with zero codebase changes |

