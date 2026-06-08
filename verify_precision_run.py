import sys, os, json, asyncio
from pathlib import Path

# Add backend package path
backend_path = Path('d:/wm_backend_/wm_backend')
sys.path.append(str(backend_path.parent))  # add d:/wm_backend_ to PYTHONPATH

# Now import the wrapper
from wm_backend.integrations.qdrant_client import QdrantClientWrapper

import httpx

QDRANT_COLLECTION = "warehouse-index"
VECTOR_SIZE = 1536
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")  # adjust port if needed

async def ensure_indexes():
    wrapper = QdrantClientWrapper()
    wrapper.ensure_collection(QDRANT_COLLECTION, VECTOR_SIZE)
    print("Indexes ensured.")

async def ingest_document():
    ingest_payload = {
        "ocr_document_id": "test-001",
        "document_type": "OCRDocument",
        "warehouse_id": "WH001",
        "text": "Dummy PDF file content for testing OCR document retrieval."
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{FASTAPI_URL}/api/rag/ingest", json=ingest_payload, timeout=30.0)
        print("Ingest response", resp.status_code, resp.text)

async def retrieve_raw_matches():
    wrapper = QdrantClientWrapper()
    client = wrapper.connect()
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    query_filter = Filter(must=[FieldCondition(key="metadata.document_type", match=MatchValue(value="OCRDocument"))])
    vector = [0.0] * VECTOR_SIZE
    results = client.query_points(collection_name=QDRANT_COLLECTION, query=vector, query_filter=query_filter, limit=10, with_payload=True)
    points = getattr(results, "points", results)
    raw_matches = len(points)
    print("raw_matches count:", raw_matches)
    top_matches = raw_matches
    print("top_matches count:", top_matches)
    for i, p in enumerate(points, 1):
        payload = getattr(p, "payload", {}) or {}
        metadata = payload.get("metadata") if "metadata" in payload else payload
        print(f"Match {i} payload:", json.dumps(metadata, indent=2))
    return raw_matches

async def call_analyze_api():
    analyze_payload = {
        "ocrDocumentId": "test-001",
        "title": "Dummy PDF file",
        "description": "What text was extracted from the uploaded document?",
        "documentType": "OCRDocument",
        "warehouseId": "WH001",
        "attemptCount": 1,
        "userId": 1
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{FASTAPI_URL}/api/ai/analyze", json=analyze_payload, timeout=30.0)
        print("Analyze API status:", resp.status_code)
        try:
            print(json.dumps(resp.json(), indent=2))
        except Exception:
            print("Response body:", resp.text)

async def main():
    await ensure_indexes()
    await ingest_document()
    await retrieve_raw_matches()
    await call_analyze_api()

if __name__ == "__main__":
    asyncio.run(main())
