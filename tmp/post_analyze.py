import sys, json, requests, time
sys.path.append('d:/rag/wm_rag')
url = 'http://127.0.0.1:8000/api/ai/analyze'
payload = {
    "ocrDocumentId": "ocr-warehouse-001",
    "title": "SKU123 location query",
    "description": "",
    "documentType": "StockReport",
    "warehouseId": "WH001",
    "attemptCount": 1,
    "userId": 1
}
# give server a moment to start
time.sleep(2)
resp = requests.post(url, json=payload)
print('Status:', resp.status_code)
print('Response JSON:')
print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
