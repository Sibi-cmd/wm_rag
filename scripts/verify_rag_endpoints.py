import os
import sys
import json
import time

# Ensure project root is in PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def run_verification():
    print("====================================================")
    print("      RAG PHASE 1 - ENDPOINT VERIFICATION           ")
    print("====================================================")

    results = {}

    # 1. GET /status
    print("\n--- 1. Testing GET /status ---")
    start = time.time()
    try:
        resp = client.get("/status")
        duration = time.time() - start
        print(f"Status Code: {resp.status_code}")
        print(f"Response: {resp.json()}")
        results["GET /status"] = {
            "status_code": resp.status_code,
            "response": resp.json(),
            "duration": f"{duration:.3f}s",
            "passed": resp.status_code == 200
        }
    except Exception as e:
        results["GET /status"] = {"status_code": "ERROR", "error": str(e), "passed": False}

    # 2. POST /api/rag/ingest
    print("\n--- 2. Testing POST /api/rag/ingest ---")
    doc_id = "test-doc-verification-111"
    ingest_payload = {
        "ocr_document_id": doc_id,
        "document_type": "WMS-AuditReport",
        "warehouse_id": "WH-TEST-VERIFY",
        "text": "This manual states that high voltage current reset procedures must be performed in Zone Z Rack R9. Under no circumstances should the operator touch auxiliary terminals before disconnecting grid power.",
        "sku": "SKU-VERIFY-99",
        "product_id": "prod-verify-99",
        "category": "High Voltage Safety Procedures",
        "zone": "Zone Z",
        "rack": "Rack R9",
        "shelf": "Shelf 1",
        "bin": "Bin A"
    }
    start = time.time()
    try:
        resp = client.post("/api/rag/ingest", json=ingest_payload)
        duration = time.time() - start
        print(f"Status Code: {resp.status_code}")
        print(f"Response: {resp.json()}")
        results["POST /api/rag/ingest"] = {
            "status_code": resp.status_code,
            "response": resp.json(),
            "duration": f"{duration:.3f}s",
            "passed": resp.status_code == 200 and resp.json().get("status") == "SUCCESS"
        }
    except Exception as e:
        results["POST /api/rag/ingest"] = {"status_code": "ERROR", "error": str(e), "passed": False}

    # 3. GET /api/rag/documents
    print("\n--- 3. Testing GET /api/rag/documents ---")
    start = time.time()
    try:
        resp = client.get("/api/rag/documents")
        duration = time.time() - start
        print(f"Status Code: {resp.status_code}")
        docs = resp.json()
        print(f"Found {len(docs)} documents.")
        # Ensure our test doc is in the list
        found = any(d.get("ocr_document_id") == doc_id for d in docs)
        results["GET /api/rag/documents"] = {
            "status_code": resp.status_code,
            "duration": f"{duration:.3f}s",
            "passed": resp.status_code == 200 and found
        }
    except Exception as e:
        results["GET /api/rag/documents"] = {"status_code": "ERROR", "error": str(e), "passed": False}

    # 4. GET /api/rag/documents/{id}
    print(f"\n--- 4. Testing GET /api/rag/documents/{doc_id} ---")
    start = time.time()
    try:
        resp = client.get(f"/api/rag/documents/{doc_id}")
        duration = time.time() - start
        print(f"Status Code: {resp.status_code}")
        doc_details = resp.json()
        print(f"Chunks Count: {len(doc_details.get('chunks', []))}")
        results[f"GET /api/rag/documents/{{id}}"] = {
            "status_code": resp.status_code,
            "response_summary": f"Chunks count: {len(doc_details.get('chunks', []))}",
            "duration": f"{duration:.3f}s",
            "passed": resp.status_code == 200 and len(doc_details.get("chunks", [])) > 0
        }
    except Exception as e:
        results[f"GET /api/rag/documents/{{id}}"] = {"status_code": "ERROR", "error": str(e), "passed": False}

    # 5. GET /api/rag/collections/stats
    print("\n--- 5. Testing GET /api/rag/collections/stats ---")
    start = time.time()
    try:
        resp = client.get("/api/rag/collections/stats")
        duration = time.time() - start
        print(f"Status Code: {resp.status_code}")
        print(f"Stats: {resp.json()}")
        results["GET /api/rag/collections/stats"] = {
            "status_code": resp.status_code,
            "response": resp.json(),
            "duration": f"{duration:.3f}s",
            "passed": resp.status_code == 200
        }
    except Exception as e:
        results["GET /api/rag/collections/stats"] = {"status_code": "ERROR", "error": str(e), "passed": False}

    # 6. POST /api/ai/analyze
    print("\n--- 6. Testing POST /api/ai/analyze ---")
    analyze_payload = {
        "ocrDocumentId": doc_id,
        "title": "Voltage safety check",
        "description": "How should I disconnect auxiliary terminals?",
        "documentType": "WMS-AuditReport",
        "warehouseId": "WH-TEST-VERIFY",
        "sku": "SKU-VERIFY-99",
        "productId": "prod-verify-99",
        "category": "High Voltage Safety Procedures",
        "zone": "Zone Z",
        "rack": "Rack R9",
        "shelf": "Shelf 1",
        "bin": "Bin A",
        "userId": 999,
        "attemptCount": 1
    }
    start = time.time()
    try:
        resp = client.post("/api/ai/analyze", json=analyze_payload)
        duration = time.time() - start
        print(f"Status Code: {resp.status_code}")
        ans = resp.json()
        print(f"Suggestion: {ans.get('suggestion')}")
        results["POST /api/ai/analyze"] = {
            "status_code": resp.status_code,
            "suggestion": ans.get("suggestion")[:100] + "...",
            "duration": f"{duration:.3f}s",
            "passed": resp.status_code == 200 and "status" in ans
        }
    except Exception as e:
        import traceback
        print(f"Analyze failed with exception: {e}")
        traceback.print_exc()
        results["POST /api/ai/analyze"] = {"status_code": "ERROR", "error": str(e), "passed": False}

    # 7. DELETE /api/rag/documents/{id}
    print(f"\n--- 7. Testing DELETE /api/rag/documents/{doc_id} ---")
    start = time.time()
    try:
        resp = client.delete(f"/api/rag/documents/{doc_id}")
        duration = time.time() - start
        print(f"Status Code: {resp.status_code}")
        print(f"Response: {resp.json()}")
        results[f"DELETE /api/rag/documents/{{id}}"] = {
            "status_code": resp.status_code,
            "response": resp.json(),
            "duration": f"{duration:.3f}s",
            "passed": resp.status_code == 200 and resp.json().get("status") == "SUCCESS"
        }
    except Exception as e:
        results[f"DELETE /api/rag/documents/{{id}}"] = {"status_code": "ERROR", "error": str(e), "passed": False}

    # 8. Post-Delete Verify GET /api/rag/documents/{id}
    print(f"\n--- 8. Re-checking GET /api/rag/documents/{doc_id} ---")
    try:
        resp = client.get(f"/api/rag/documents/{doc_id}")
        print(f"Status Code after delete (should be 404): {resp.status_code}")
        results["GET /api/rag/documents/{id} after DELETE"] = {
            "status_code": resp.status_code,
            "passed": resp.status_code == 404
        }
    except Exception as e:
        results["GET /api/rag/documents/{id} after DELETE"] = {"status_code": "ERROR", "error": str(e), "passed": False}

    print("\n====================================================")
    print("               VERIFICATION MATRIX SUMMARY          ")
    print("====================================================")
    for endpoint, detail in results.items():
        pass_str = "PASS" if detail["passed"] else "FAIL"
        print(f"{endpoint:<35} | Status: {detail.get('status_code'):<5} | Duration: {detail.get('duration', 'N/A'):<7} | {pass_str}")

if __name__ == "__main__":
    run_verification()
