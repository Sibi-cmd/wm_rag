"""
Test query script — standalone testing of the RAG pipeline.
Usage: python scripts/test_query.py "your question here" [document_type]
"""
import os
import sys
import asyncio
import json
from dotenv import load_dotenv

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag_pipeline import RAGPipeline

load_dotenv()

async def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/test_query.py "your question here" [document_type]')
        sys.exit(1)
        
    query = sys.argv[1]
    document_type = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"\nInitializing RAG Pipeline for query: '{query}'")
    if document_type:
        print(f"Filtering by document type: '{document_type}'")
        
    pipeline = RAGPipeline()
    
    print("\n--- Phase 1: Retrieve & Rerank ---")
    # Use the core pipeline components directly
    query_vector = pipeline.core_pipeline.embedder.embed(query)
    
    filters = {}
    if document_type:
        filters["metadata.document_type"] = document_type
    
    raw_results = pipeline.core_pipeline.vector_store.search(
        vector=query_vector,
        top_k=8,
        filters=filters
    )
    print(f"Retrieved {len(raw_results)} raw chunks from Qdrant.")
    
    reranked_results = pipeline.core_pipeline.reranker.rerank(query, raw_results, top_k=4)
    print(f"Reranked and kept top {len(reranked_results)} chunks:")
    for idx, result in enumerate(reranked_results):
        meta = result.metadata or {}
        print(f"  [{idx+1}] Score: {result.score:.4f} | Section: {meta.get('section')} | Document ID: {meta.get('ocr_document_id')}")
        text_snippet = result.text[:120].replace('\n', ' ')
        print(f"      Text: {text_snippet}...")
        
    print("\n--- Phase 2: Generating Response using Gemini ---")
    try:
        suggestion, confidence, category, ocr_doc_id = await pipeline.execute(
            query=query,
            document_type=document_type
        )
        
        print("\n--- RESULTS ---")
        print(f"Predicted Category:   {category}")
        print(f"Confidence Score:     {confidence:.4f}")
        print(f"Retrieved Document ID: {ocr_doc_id}")
        print("\nGenerated Suggestion (JSON):")
        try:
            parsed = json.loads(suggestion)
            print(json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            print(suggestion)
            
    except Exception as e:
        print(f"Error during generation: {e}")

if __name__ == "__main__":
    asyncio.run(main())
