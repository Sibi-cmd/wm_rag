import sys, os, json
sys.path.append('d:/rag/wm_rag')

from fastapi.testclient import TestClient
from app.main import app
from app.database import get_qdrant_client

def recreate_and_index():
    client = get_qdrant_client()
    # drop and create collection
    try:
        client.delete_collection(collection_name='warehouse-index')
    except Exception:
        pass
    client.recreate_collection(
        collection_name='warehouse-index',
        vectors_config={"default": {"size": 768, "distance": "Cosine"}}
    )
    # create payload index for document_type
    try:
        client.create_payload_index(
            collection_name='warehouse-index',
            field_name='metadata.document_type',
            field_schema='keyword'
        )
    except Exception as e:
        print('Index creation error:', e)

def run_test():
    payload = {
        "ocrDocumentId": "ocr-warehouse-001",
        "title": "SKU123 location query",
        "description": "",
        "documentType": "StockReport",
        "priority": None,
        "warehouseId": "WH001",
        "sku": None,
        "productId": None,
        "category": None,
        "zone": None,
        "rack": None,
        "bin": None,
        "auditTrail": [],
        "attemptCount": 1,
        "previousResponse": None,
        "followUpQuestion": None,
        "userId": 1
    }
    client = TestClient(app)
    resp = client.post('/api/ai/analyze', json=payload)
    print('Status:', resp.status_code)
    print('Response JSON:')
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))

if __name__ == '__main__':
    recreate_and_index()
    run_test()
