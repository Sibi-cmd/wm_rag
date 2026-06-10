"""
Refresh (delete and recreate) the Qdrant collection index.
Usage: python scripts/refresh_index.py
"""
import os
import sys
from dotenv import load_dotenv

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import get_qdrant_client
from qdrant_client.models import Distance, VectorParams

load_dotenv()

def refresh_index(collection_name: str = "warehouse-index"):
    client = get_qdrant_client()
    
    print(f"Refreshing Qdrant collection index: '{collection_name}'...")
    try:
        # Delete existing collection if present
        try:
            client.delete_collection(collection_name=collection_name)
            print(f"Deleted existing collection '{collection_name}'.")
        except Exception:
            pass  # Collection may not exist

        # Create fresh collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        print(f"Collection '{collection_name}' successfully created with 768-dim COSINE distance config.")
    except Exception as e:
        print(f"Error refreshing collection: {e}")
        sys.exit(1)

if __name__ == "__main__":
    refresh_index()
