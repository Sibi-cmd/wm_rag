from typing import List, Optional
from .database import get_qdrant_client
from .embeddings import get_embedding
from qdrant_client.models import Filter, FieldCondition, MatchValue

class QdrantRetriever:
    def __init__(self, collection_name: str = "ev-manual-index"):
        self.client = get_qdrant_client()
        self.collection_name = collection_name

    def retrieve(self, query: str, top_k: int = 8, vehicle_model: Optional[str] = None) -> List[dict]:
        """
        Query Qdrant collection for matching documents.
        Optionally filter by vehicle_model.
        """
        try:
            vector = get_embedding(query)
            
            # Construct a Qdrant query filter if vehicle_model is provided
            query_filter = None
            if vehicle_model:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="metadata.vehicle_model",
                            match=MatchValue(value=vehicle_model)
                        )
                    ]
                )
            
            # Search the Qdrant index
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True
            )
            
            # If no results found with model filter, fallback to querying without the filter
            if not results and vehicle_model:
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=vector,
                    limit=top_k,
                    with_payload=True
                )
                
            # Standardize output to match Pinecone output structure for seamless transition
            matches = []
            for item in results:
                payload = item.payload or {}
                # Handle potential structures: nested "metadata" field or flat payload
                metadata = payload.get("metadata") if "metadata" in payload else payload
                matches.append({
                    "id": item.id,
                    "score": item.score,
                    "metadata": metadata
                })
            return matches
        except Exception as e:
            print(f"Error querying Qdrant: {e}", flush=True)
            return []
