import os
import sys
import uuid
import tempfile
from datetime import datetime
from dotenv import load_dotenv

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import get_qdrant_client, get_mongodb_db
from app.s3_handler import S3Handler
from app.processor import DocumentProcessor
from app.embeddings import get_embedding
from qdrant_client.models import PointStruct

load_dotenv()

def clear_qdrant_collection(collection_name="ev-manual-index"):
    print(f"\n--- Clearing existing Qdrant collection: '{collection_name}' ---")
    try:
        from scripts.refresh_index import refresh_index
        refresh_index(collection_name)
    except Exception as e:
        print(f"Error clearing Qdrant collection: {e}")

def ingest_documents(
    s3_key: str = None,
    file_path: str = None,
    vehicle_model: str = "Unknown",
    clear_existing: bool = False,
    collection_name: str = "ev-manual-index"
):
    processor = DocumentProcessor()
    qdrant_client = get_qdrant_client()
    mongo_db = get_mongodb_db()

    if clear_existing:
        clear_qdrant_collection(collection_name)

    temp_path = None
    target_path = None

    try:
        if file_path:
            if not os.path.exists(file_path):
                # Check inside data/raw as fallback
                fallback_path = os.path.join("data", "raw", os.path.basename(file_path))
                if os.path.exists(fallback_path):
                    target_path = fallback_path
                else:
                    raise FileNotFoundError(f"Local file not found: {file_path}")
            else:
                target_path = file_path
            print(f"Processing local file: '{target_path}'")
        elif s3_key:
            print(f"Downloading '{s3_key}' from S3...")
            s3 = S3Handler()
            # Determine suffix
            suffix = os.path.splitext(s3_key)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                temp_path = tmp_file.name
            
            downloaded = s3.download_file(s3_key, temp_path)
            if not downloaded:
                raise RuntimeError(f"Failed to download '{s3_key}' from S3")
            target_path = temp_path
        else:
            raise ValueError("Either 'file_path' or 's3_key' must be specified.")

        # Extract & Chunk
        print("Extracting and semantic chunking document...")
        chunks = processor.extract_and_chunk(target_path)
        print(f"Total chunks extracted: {len(chunks)}")

        if not chunks:
            print("No chunks produced. Exiting.")
            return

        # Prepare and upsert points
        print(f"\nProcessing and embedding {len(chunks)} chunks...")
        points = []
        
        for i, chunk in enumerate(chunks):
            text = chunk['text']
            embedding = get_embedding(text)
            
            # Create a deterministic UUID from chunk_id
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk['chunk_id']))
            
            metadata = {
                "chunk_id": chunk['chunk_id'],
                "issue_id": chunk['issue_id'],
                "vehicle_model": vehicle_model,
                "source_file": s3_key or os.path.basename(target_path),
                "section": chunk['section'],
                "severity": chunk['severity'],
                "escalation_required": chunk['escalation_required'],
                "text": text,
                "chunk_index": chunk['chunk_index']
            }

            # Upsert into MongoDB for local chunk tracking
            try:
                mongo_db.chunks.update_one(
                    {"chunk_id": chunk['chunk_id']},
                    {"$set": {**metadata, "updated_at": datetime.utcnow()}},
                    upsert=True
                )
            except Exception as e:
                # Silently catch MongoDB logging issues during script run
                pass

            points.append(
                PointStruct(
                    id=point_uuid,
                    vector=embedding,
                    payload={"metadata": metadata}
                )
            )

            # Upsert in batches of 50
            if len(points) >= 50:
                qdrant_client.upsert(collection_name=collection_name, points=points)
                print(f"  Upserted {i+1}/{len(chunks)} chunks to Qdrant...")
                points = []

        if points:
            qdrant_client.upsert(collection_name=collection_name, points=points)
            print(f"  Upserted remaining chunks to Qdrant...")

        print(f"\nIngestion complete! {len(chunks)} chunks successfully indexed in Qdrant & MongoDB.")

    except Exception as e:
        print(f"ERROR during ingestion: {e}")
        raise
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/ingest_documents.py <file_path_or_s3_key> <vehicle_model> [clear_existing=False]")
        sys.exit(1)
    
    source = sys.argv[1]
    model = sys.argv[2]
    clear = False
    if len(sys.argv) > 3:
        clear = sys.argv[3].lower() in ['true', '1', 'yes']

    # Determine if source is local or S3 key
    if os.path.exists(source) or source.startswith("data/") or "/" in source or "\\" in source:
        ingest_documents(file_path=source, vehicle_model=model, clear_existing=clear)
    else:
        ingest_documents(s3_key=source, vehicle_model=model, clear_existing=clear)
