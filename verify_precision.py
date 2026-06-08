import sys, os, json, asyncio

# Add project root to sys.path
PROJECT_ROOT = r"d:/rag/wm_rag"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from app.integrations.qdrant_client import QdrantClientWrapper
from app.retriever import QdrantRetriever
import httpx

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
COLLECTION = "warehouse-index"
VECTOR_SIZE = 1536

async def ensure_indexes():
    wrapper = QdrantClientWrapper()
    wrapper.ensure_collection(COLLECTION, VECTOR_SIZE)
    print("✅ Indexes ensured")

async def ingest_document():
    payload = {
        "ocr_document_id": "test-001",
        "document_type": "OCRDocument",
        "warehouse_id": "WH001",
        "text": "Dummy PDF file content for testing OCR document retrieval."
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{FASTAPI_URL}/api/rag/ingest", json=payload)
        print("Ingest response:", resp.status_code)
        print(resp.text)

async def retrieve_raw_matches():
    wrapper = QdrantClientWrapper()
    client = wrapper.connect()
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    query_filter = Filter(must=[FieldCondition(key="metadata.document_type", match=MatchValue(value="OCRDocument"))])
    vector = [0.0] * VECTOR_SIZE
    results = client.query_points(collection_name=COLLECTION, query=vector, query_filter=query_filter, limit=8, with_payload=True)
    points = getattr(results, "points", results)
    raw = len(points)
    print("raw_matches count:", raw)
    for i, p in enumerate(points, 1):
        payload = getattr(p, "payload", {}) or {}
        metadata = payload.get("metadata") if "metadata" in payload else payload
        print(f"Match {i} payload:", json.dumps(metadata, indent=2))
    return raw

async def call_analyze_api():
    payload = {
        "ocrDocumentId": "test-001",
        "title": "Dummy PDF file",
        "description": "What text was extracted from the uploaded document?",
        "documentType": "OCRDocument",
        "warehouseId": "WH001",
        "attemptCount": 1,
        "userId": 1
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{FASTAPI_URL}/api/ai/analyze", json=payload)
        print("Analyze API status:", resp.status_code)
        print("Response body:\n", json.dumps(resp.json(), indent=2))

async def main():
    await ensure_indexes()
    await ingest_document()
    raw = await retrieve_raw_matches()
    await call_analyze_api()
    print("\n✅ Verification completed. raw_matches =", raw)

if __name__ == "__main__":
    asyncio.run(main())
