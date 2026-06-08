from typing import List, Optional
import logging
from .database import get_qdrant_client
from .embeddings import get_embedding
from qdrant_client.models import Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)

class QdrantRetriever:
    def __init__(self, collection_name: str = "warehouse-index"):
        self.client = get_qdrant_client()
        self.collection_name = collection_name

    def retrieve(self, query: str, top_k: int = 8, document_type: Optional[str] = None) -> List[dict]:
        """
        Query Qdrant collection for matching documents.
        Optionally filter by document_type.
        """
        try:
            vector = get_embedding(query)
            
            # Construct a Qdrant query filter if document_type is provided
            query_filter = None
            if document_type:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="metadata.document_type",
                            match=MatchValue(value=document_type)
                        )
                    ]
                )
            
            # Search the Qdrant index
            try:
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=vector,
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True
                )
            except Exception as e:
                # If filter causes a bad request (missing index), fallback to query_points without filter
                logger.warning(f"query_points failed ({e}), retrying without filter", flush=True)
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=vector,
                    limit=top_k,
                    with_payload=True
                )
            
            # Extract points list from QueryResponse (new client version)
            points = getattr(results, "points", results)
            
            # If query_points returned no points and a document_type filter was used, fallback to older search API
            if not points and document_type:
                # If query_points returned no points and a document_type filter was used, retry query_points without filter
                points = self.client.query_points(
                    collection_name=self.collection_name,
                    query=vector,
                    limit=top_k,
                    with_payload=True
                )
                
            # Standardize output to match Pinecone output structure for seamless transition
            matches = []
            for item in points:
                payload = getattr(item, "payload", {}) or {}
                # Handle potential structures: nested "metadata" field or flat payload
                metadata = payload.get("metadata") if "metadata" in payload else payload
                matches.append({
                    "id": getattr(item, "id", None),
                    "score": getattr(item, "score", None),
                    "metadata": metadata
                })
            return matches
        except Exception as e:
            print(f"Error querying Qdrant: {e}", flush=True)
            return []
