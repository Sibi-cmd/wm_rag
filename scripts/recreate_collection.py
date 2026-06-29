"""
Recreate the Qdrant collection with proper configuration.
Usage: python scripts/recreate_collection.py
"""
import sys
import os

# Ensure project root is in PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.database import get_qdrant_client
from qdrant_client.models import VectorParams, Distance

COLLECTION_NAME = "warehouse-index"
VECTOR_SIZE = 768  # Correct dimension for the all-mpnet-base-v2 embedding model

def recreate_collection():
    client = get_qdrant_client()

    # Delete if exists
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'.")
    except Exception as e:
        print(f"Delete skipped (collection may not exist): {e}")

    # Create with proper VectorParams config
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )

    # Add payload index for document_type (keyword) to enable filtering
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="metadata.document_type",
        field_schema="keyword"
    )
    # Add payload index for ocr_document_id (keyword) to enable filtering & deletions
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="metadata.ocr_document_id",
        field_schema="keyword"
    )
    # Add payload indexes for other metadata filters
    for field in ["sku", "warehouse_id", "product_id", "category", "zone", "rack", "shelf", "bin"]:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=f"metadata.{field}",
            field_schema="keyword"
        )
    print(f"Recreated collection '{COLLECTION_NAME}' with vector size {VECTOR_SIZE}, COSINE distance, and all keyword payload indexes.")

if __name__ == "__main__":
    recreate_collection()
