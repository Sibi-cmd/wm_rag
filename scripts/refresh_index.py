import os
import sys
from dotenv import load_dotenv

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import get_qdrant_client
from qdrant_client.models import Distance, VectorParams

load_dotenv()

def refresh_index(collection_name: str = "ev-manual-index"):
    client = get_qdrant_client()
    
    print(f"Re-creating/Refresing Qdrant collection index: '{collection_name}'...")
    try:
        # Recreate collection deletes if exists, then creates
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        print(f"Collection '{collection_name}' successfully created/refreshed with 768-dim COSINE distance config.")
    except Exception as e:
        print(f"Error recreating collection: {e}")
        sys.exit(1)

if __name__ == "__main__":
    refresh_index()
