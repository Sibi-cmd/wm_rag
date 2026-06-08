import sys, json
sys.path.append('d:/rag/wm_rag')
from fastapi.testclient import TestClient
from app.main import app

payload = {
    "ocrDocumentId": "ocr-warehouse-001",
    "title": "SKU123 location query",
    "description": "",
    "documentType": "StockReport",
    "warehouseId": "WH001",
    "attemptCount": 1,
    "userId": 1,
    "priority": None,
    "sku": None,
    "productId": None,
    "category": None,
    "zone": None,
    "rack": None,
    "bin": None,
    "auditTrail": [],
    "previousResponse": None,
    "followUpQuestion": None
}
client = TestClient(app)
response = client.post("/api/ai/analyze", json=payload)
print("Status:", response.status_code)
print("Response JSON:")
print(json.dumps(response.json(), indent=2, ensure_ascii=False))
