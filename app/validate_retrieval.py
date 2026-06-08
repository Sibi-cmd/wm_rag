import asyncio, json, os
import httpx
from wm_rag.app.integrations.qdrant_client import QdrantClientWrapper

# Configuration
QDRANT_COLLECTION = "warehouse-index"
VECTOR_SIZE = 1536  # assumed embedding size
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")

async def ensure_indexes():
    wrapper = QdrantClientWrapper()
    wrapper.ensure_collection(QDRANT_COLLECTION, VECTOR_SIZE)
    print("Ensured collection and payload indexes.")

async def ingest_document():
    ingest_payload = {
        "ocr_document_id": "test-001",
        "document_type": "OCRDocument",
        "warehouse_id": "WH001",
        "text": "Dummy PDF file content for testing OCR document retrieval."
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{FASTAPI_URL}/api/rag/ingest", json=ingest_payload, timeout=30.0)
        print("Ingest response:", response.status_code, response.text)

async def retrieve_raw_matches():
    wrapper = QdrantClientWrapper()
    client = wrapper.connect()
    # Build filter
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    query_filter = Filter(must=[FieldCondition(key="metadata.document_type", match=MatchValue(value="OCRDocument"))])
    # Use a zero vector (all zeros) since we only need filter results; vector size must match
    vector = [0.0] * VECTOR_SIZE
    results = client.query_points(collection_name=QDRANT_COLLECTION, query=vector, query_filter=query_filter, limit=8, with_payload=True)
    points = getattr(results, "points", results)
    raw_matches = len(points)
    print("raw_matches count:", raw_matches)
    # top_matches count is same as raw_matches in this simple test
    print("top_matches count:", raw_matches)
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
        print("Analyze API response status:", resp.status_code)
        print("Analyze API response body:")
        print(json.dumps(resp.json(), indent=2))

async def main():
    await ensure_indexes()
    await ingest_document()
    await retrieve_raw_matches()
    await call_analyze_api()

if __name__ == "__main__":
    asyncio.run(main())
