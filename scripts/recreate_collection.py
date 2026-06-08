import sys
import os
# Ensure project root is in PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.database import get_qdrant_client
from qdrant_client.http.models import VectorParams, Distance, CollectionConfig

COLLECTION_NAME = "warehouse-index"
VECTOR_SIZE = 768  # Correct dimension for the embedding model

def recreate_collection():
    client = get_qdrant_client()
    # Delete if exists
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'.")
    except Exception as e:
        print(f"Delete failed (might not exist): {e}")

    # Create with proper config
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"": {"size": VECTOR_SIZE, "distance": "Cosine"}}
    )
    # Add payload index for document_type (keyword) to enable filtering
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="metadata.document_type",
        field_schema="keyword"
    )
    print(f"Recreated collection '{COLLECTION_NAME}' with vector size {VECTOR_SIZE}.")

if __name__ == "__main__":
    recreate_collection()
