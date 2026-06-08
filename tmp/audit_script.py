import os, sys, json
sys.path.append('d:/rag/wm_rag')
from app.retriever import QdrantRetriever
retr = QdrantRetriever()
# Use a dummy zero vector of correct dim
vector = [0.0] * 768
result = retr.client.query_points(collection_name='warehouse-index', query=vector, limit=5, with_payload=True)
output = {
    'type_result': str(type(result)),
    'len_result': len(result) if hasattr(result, '__len__') else None,
    'first_type': str(type(result[0])) if result else None,
    'first_repr': repr(result[0]) if result else None,
    'first_attrs': {attr: getattr(result[0], attr) for attr in dir(result[0]) if not attr.startswith('_')} if result else None
}
with open('d:/rag/wm_rag/tmp/audit_output.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, default=str)
print('audit written')
