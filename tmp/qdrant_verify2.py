import os, sys
from qdrant_client import QdrantClient

url = os.getenv('QDRANT_URL')
api_key = os.getenv('QDRANT_API_KEY')

def report(step, success, detail=''):
    print(f'{step}:', 'PASS' if success else 'FAIL', f'| {detail}')

# 1. Connect
try:
    client = QdrantClient(url=url, api_key=api_key)
    # simple call to ensure connection
    client.get_collections()
    report('Qdrant connectivity', True)
except Exception as e:
    report('Qdrant connectivity', False, str(e))
    sys.exit(0)

# 2. List collections
try:
    coll_list = client.get_collections().collections
    names = [c.name for c in coll_list]
    report('List collections', True, f'found {len(names)}')
except Exception as e:
    report('List collections', False, str(e))
    names = []

# 3. Ensure warehouse-index exists (create if missing)
collection_name = 'warehouse-index'
if collection_name in names:
    report('warehouse-index existence', True)
else:
    try:
        client.recreate_collection(collection_name=collection_name,
                                 vectors_config={'size': 768, 'distance': 'Cosine'})
        report('warehouse-index creation', True)
    except Exception as e:
        report('warehouse-index creation', False, str(e))
        sys.exit(0)

# 4. Verify dimension
try:
    info = client.get_collection(collection_name)
    dim = info.config.params.vector_size
    report('Vector dimension', dim == 768, f'found {dim}')
except Exception as e:
    report('Vector dimension', False, str(e))

# 5. Retriever init
try:
    from app.retriever import QdrantRetriever
    retriever = QdrantRetriever(collection_name)
    report('Retriever init', True)
except Exception as e:
    report('Retriever init', False, str(e))

print('Verification completed')
