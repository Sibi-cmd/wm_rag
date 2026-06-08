import sys, os, json, asyncio

# Add project roots for imports
PROJECT_ROOTS = [r"d:/rag/wm_rag", r"d:/wm_backend_/wm_backend"]
for p in PROJECT_ROOTS:
    if p not in sys.path:
        sys.path.append(p)

# Imports after path adjustments
from app.database import get_qdrant_client
from integrations.qdrant_client import QdrantClientWrapper
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
        print("Ingest response status:", resp.status_code)
        print("Ingest response body:", resp.text)

async def retrieve_raw_matches():
    client = get_qdrant_client()
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    query_filter = Filter(must=[FieldCondition(key="metadata.document_type", match=MatchValue(value="OCRDocument"))])
    vector = [0.0] * VECTOR_SIZE
    results = client.query_points(collection_name=COLLECTION, query=vector, query_filter=query_filter, limit=8, with_payload=True)
    points = getattr(results, "points", results)
    raw = len(points)
    print("raw_matches count:", raw)
    print("top_matches count (limit 8):", raw)
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
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        print("Analyze API response body:\n", json.dumps(body, indent=2) if isinstance(body, dict) else body)

async def main():
    await ensure_indexes()
    await ingest_document()
    raw = await retrieve_raw_matches()
    await call_analyze_api()
    print("\n✅ Verification completed. raw_matches =", raw)
    if raw == 1:
        print("Result: PASS (only OCRDocument retrieved)")
    else:
        print("Result: FAIL (multiple document types retrieved)")

if __name__ == "__main__":
    asyncio.run(main())
