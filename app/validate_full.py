import asyncio
import json
from wm_rag.app.main import rag_ingest, analyze_query
from wm_rag.app.models import IngestRequest, WarehouseRequest
from wm_rag.app.rag_pipeline import RAGPipeline

async def main():
    # Ingest test document
    ingest_payload = IngestRequest(
        ocr_document_id="test-001",
        document_type="OCRDocument",
        warehouse_id="WH001",
        text="Dummy PDF file content for testing retrieval."
    )
    ingest_result = await rag_ingest(ingest_payload)
    print("Ingest result:", ingest_result)

    # Perform retrieval via analyzer (which uses retriever internally)
    request_payload = WarehouseRequest(
        ocrDocumentId="test-001",
        title="Dummy PDF file",
        description="What text was extracted from the uploaded document?",
        documentType="OCRDocument",
        warehouseId="WH001",
        attemptCount=1,
        userId=1,
        # other required fields with defaults if any
        followUpQuestion=None,
        priority=None,
        previousResponse=None,
        auditTrail=[]
    )
    response = await analyze_query(request_payload)
    print("API response:", response)

    # Direct retrieval using retriever for detailed output
    from wm_rag.app.retriever import QdrantRetriever
    retriever = QdrantRetriever()
    matches = retriever.retrieve("Dummy PDF file", document_type="OCRDocument")
    print("raw_matches count:", len(matches))
    print("matches payloads:", json.dumps(matches, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
