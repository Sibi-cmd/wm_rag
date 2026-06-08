import os, sys, json
# Ensure project root is in PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
payload = {
    "ocr_document_id": "ocr-warehouse-001",
    "document_type": "StockReport",
    "warehouse_id": "WH001",
    "text": "Product SKU123 is stored in Zone A Rack R2 Bin B5. Current quantity is 250 units. Reorder level is 50 units."
}
resp = client.post('/api/rag/ingest', json=payload)
print(json.dumps({"status_code": resp.status_code, "body": resp.json()}, indent=2))
