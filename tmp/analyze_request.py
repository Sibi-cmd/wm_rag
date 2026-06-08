import httpx, json, sys
payload = {
    "ocrDocumentId": "ocr-warehouse-001",
    "title": "SKU123 location query",
    "description": "",
    "documentType": "StockReport",
    "warehouseId": "WH001",
    "attemptCount": 1,
    "userId": 1
}
try:
    resp = httpx.post('http://127.0.0.1:8000/api/ai/analyze', json=payload, timeout=30.0)
    print('STATUS', resp.status_code)
    print('BODY', resp.text)
except Exception as e:
    print('ERROR', e, file=sys.stderr)
    sys.exit(1)
