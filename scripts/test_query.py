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
        print("Usage: python scripts/test_query.py \"<your question here>\" [vehicle_model]")
        sys.exit(1)
        
    query = sys.argv[1]
    vehicle_model = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"\nInitializing RAG Pipeline for query: '{query}'")
    if vehicle_model:
        print(f"Filtering by vehicle model: '{vehicle_model}'")
        
    pipeline = RAGPipeline()
    
    print("\n--- Phase 1: Retrieve & Rerank ---")
    raw_matches = pipeline.retriever.retrieve(query, top_k=8, vehicle_model=vehicle_model)
    print(f"Retrieved {len(raw_matches)} raw chunks from Qdrant.")
    
    reranked_matches = pipeline.reranker.rerank(query, raw_matches, top_n=4)
    print(f"Reranked and kept top {len(reranked_matches)} chunks:")
    for idx, match in enumerate(reranked_matches):
        meta = match.get("metadata", {})
        print(f"  [{idx+1}] Score: {match['score']:.4f} | Section: {meta.get('section')} | ID: {meta.get('issue_id')}")
        text_snippet = meta.get('text', '')[:120].replace('\n', ' ')
        print(f"      Text: {text_snippet}...")
        
    print("\n--- Phase 2: Generating Response using Gemini ---")
    try:
        suggestion, confidence, category, issue_id = await pipeline.execute(
            query=query,
            vehicle_model=vehicle_model
        )
        
        print("\n--- RESULTS ---")
        print(f"Predicted Category: {category}")
        print(f"Confidence Score:   {confidence:.4f}")
        print(f"Retrieved Issue ID: {issue_id}")
        print("\nGenerated Suggestion (JSON):")
        try:
            parsed = json.loads(suggestion)
            print(json.dumps(parsed, indent=2))
        except:
            print(suggestion)
            
    except Exception as e:
        print(f"Error during generation: {e}")

if __name__ == "__main__":
    asyncio.run(main())
