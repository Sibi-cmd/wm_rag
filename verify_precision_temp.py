import os, json, sys
import httpx

# Ensure indexes using QdrantClientWrapper
try:
    from app.integrations.qdrant_client import QdrantClientWrapper
except Exception as e:
    print('Import error:', e)
    sys.exit(1)

wrapper = QdrantClientWrapper()
collection = "warehouse-index"
vector_size = 1536
wrapper.ensure_collection(collection, vector_size)
print('Indexes ensured')

# Build filter query for OCRDocument
filter_payload = {
    "must": [
        {
            "key": "metadata.document_type",
            "match": {"value": "OCRDocument"}
        }
    ]
}
# Zero vector
vector = [0.0] * vector_size
search_body = {
    "vector": vector,
    "filter": filter_payload,
    "limit": 10,
    "with_payload": True,
    "with_vector": False
}
# Qdrant HTTP endpoint (adjust if different host/port)
qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
search_url = f"{qdrant_url}/collections/{collection}/points/search"
resp = httpx.post(search_url, json=search_body, timeout=30.0)
if resp.status_code != 200:
    print('Search API error', resp.status_code, resp.text)
    sys.exit(1)
result = resp.json()
points = result.get('result', [])
raw_matches = len(points)
print('raw_matches count:', raw_matches)
print('top_matches count:', raw_matches)
for i, pt in enumerate(points, 1):
    payload = pt.get('payload', {})
    metadata = payload.get('metadata', payload)
    print(f"Match {i} payload:", json.dumps(metadata, indent=2))

# Call the FastAPI analyze endpoint
analyze_payload = {
    "ocrDocumentId": "test-001",
    "title": "Dummy PDF file",
    "description": "What text was extracted from the uploaded document?",
    "documentType": "OCRDocument",
    "warehouseId": "WH001",
    "attemptCount": 1,
    "userId": 1
}
fastapi_url = os.getenv("FASTAPI_URL", "http://127.0.0.1:8001")
api_url = f"{fastapi_url}/api/ai/analyze"
resp2 = httpx.post(api_url, json=analyze_payload, timeout=30.0)
print('API response status:', resp2.status_code)
print('API response body:', json.dumps(resp2.json(), indent=2))
