import os, sys
sys.path.append('d:/rag/wm_rag')
from app.retriever import QdrantRetriever

retr = QdrantRetriever()
result = retr.client.query_points(collection_name='warehouse-index', query=[0]*768, limit=5, with_payload=True)
print('Result type:', type(result))
if result:
    first = result[0]
    print('First element type:', type(first))
    # Print dict of attributes
    attrs = {attr: getattr(first, attr) for attr in dir(first) if not attr.startswith('_')}
    print('First element attrs:', attrs)
else:
    print('Result is empty')
