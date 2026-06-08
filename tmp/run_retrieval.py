import json, sys
sys.path.append('d:/rag/wm_rag')
from app.retriever import QdrantRetriever
retr = QdrantRetriever()
matches = retr.retrieve('Where is SKU123 stored?', top_k=5)
print(json.dumps(matches, ensure_ascii=False, indent=2))
