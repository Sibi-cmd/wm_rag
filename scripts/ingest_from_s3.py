import os
import sys
import uuid
import tempfile
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.s3_handler import S3Handler
from app.processor import DocumentProcessor

load_dotenv()

# Lazy-loaded Embeddings Model
_embedding_model = None

def get_embeddings_model():
    global _embedding_model
    model_name = "sentence-transformers/all-mpnet-base-v2"
    if _embedding_model is None:
        print(f"Loading SentenceTransformer model ({model_name})...")
        _embedding_model = SentenceTransformer(model_name)
        print("Model loaded.")
    return _embedding_model

def clear_pinecone_data(pinecone_index):
    print("\n--- Clearing existing Pinecone data ---")
    try:
        pinecone_index.delete(delete_all=True)
        print("  Pinecone: All vectors deleted from index.")
    except Exception as e:
        print(f"  Error clearing Pinecone index: {e}")
    print("--- Cleanup complete ---\n")

def ingest_from_s3(s3_key, vehicle_model, clear_existing=False):
    # --- Connections ---
    s3 = S3Handler()
    processor = DocumentProcessor()

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is not set in .env")
    
    pc = Pinecone(api_key=pinecone_api_key)
    pinecone_index = pc.Index("ev-manual-index")
    print("Pinecone connected to index: ev-manual-index")

    if clear_existing:
        clear_pinecone_data(pinecone_index)

    # --- Download from S3 ---
    print(f"\nDownloading '{s3_key}' from S3...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(s3_key)[1]) as tmp_file:
        tmp_path = tmp_file.name

    try:
        downloaded_path = s3.download_file(s3_key, tmp_path)
        if not downloaded_path:
            print(f"ERROR: Failed to download '{s3_key}' from S3.")
            return

        # --- Extract & Chunk ---
        print("Extracting and semantic chunking document...")
        chunks = processor.extract_and_chunk(downloaded_path)
        print(f"Total chunks extracted: {len(chunks)}")

        if not chunks:
            print("No chunks produced. Exiting.")
            return

        # --- Load embedding model ---
        model = get_embeddings_model()

        # --- Step 1: Generate embeddings and prepare batches ---
        print(f"\nProcessing {len(chunks)} chunks...")
        pinecone_batch = []
        
        for i, chunk in enumerate(chunks):
            text = chunk['text']
            embedding = model.encode(text).tolist()
            
            pinecone_batch.append({
                "id": chunk['chunk_id'],
                "values": embedding,
                "metadata": {
                    "issue_id": chunk['issue_id'],
                    "vehicle_model": vehicle_model,
                    "source_file": s3_key,
                    "section": chunk['section'],
                    "severity": chunk['severity'],
                    "escalation_required": chunk['escalation_required'],
                    "text": text,
                    "chunk_index": chunk['chunk_index']
                }
            })

            # Batch upsert every 50 items
            if len(pinecone_batch) >= 50:
                pinecone_index.upsert(vectors=pinecone_batch)
                print(f"  Upserted {i+1}/{len(chunks)} chunks to Pinecone...")
                pinecone_batch = []

        # Flush remaining
        if pinecone_batch:
            pinecone_index.upsert(vectors=pinecone_batch)

        print(f"\nIngestion complete! {len(chunks)} chunks stored in Pinecone successfully.")

    except Exception as e:
        print(f"\nERROR during ingestion: {e}")
        raise
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/ingest_from_s3.py <s3_key> <vehicle_model>")
        sys.exit(1)
    
    ingest_from_s3(sys.argv[1], sys.argv[2])
