import sys, os, json
sys.path.append('d:/rag/wm_rag')

from app.rag_pipeline import RAGPipeline
from app.retriever import QdrantRetriever

# Parameters
query = "Where is SKU123 stored?"
document_type = "StockReport"

# 1. Direct retrieval debug
retriever = QdrantRetriever()
raw_matches = retriever.retrieve(query, top_k=8, document_type=document_type)
print('=== Direct Retrieval ===')
print(f'Retrieved chunk count: {len(raw_matches)}')
for i, m in enumerate(raw_matches, 1):
    meta = m.get('metadata', {})
    text = meta.get('text', '<no text>')
    print(f'Chunk {i} metadata text: {text}')

# 2. Full RAG pipeline
pipeline = RAGPipeline()
suggestion, confidence, category, ocr_id = await pipeline.execute(
    query=query,
    document_type=document_type,
    audit_trail_text="",
    attempt_count=1,
    prev_response=None,
    follow_up_question=None,
    priority=None
)
print('\n=== RAG Pipeline Result ===')
print('Suggestion JSON:', suggestion)
print('Confidence:', confidence)
print('Predicted Category:', category)
print('Top OCR Doc ID:', ocr_id)
