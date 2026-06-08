import os, json, sys
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

# Load configuration from environment (already loaded via dotenv in app)
url = os.getenv('QDRANT_URL')
api_key = os.getenv('QDRANT_API_KEY')

def report(step, success, detail=''):
    print(f'{step}:', 'PASS' if success else 'FAIL', f'| {detail}')

# 1. Qdrant authentication & connectivity
try:
    client = QdrantClient(url=url, api_key=api_key)
    # simple health check
    client.get_collection("__nonexistent__", timeout=5)
    report('Qdrant connectivity', True)
except Exception as e:
    report('Qdrant connectivity', False, str(e))
    sys.exit(0)

# 2. List existing collections
try:
    coll_list = client.get_collections().collections
    names = [c.name for c in coll_list]
    report('List collections', True, f'found {len(names)} collections')
except Exception as e:
    report('List collections', False, str(e))
    names = []

# 3-5. Check/create warehouse-index
collection_name = 'warehouse-index'
if collection_name in names:
    report('warehouse-index existence', True)
else:
    try:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config={'size': 768, 'distance': 'Cosine'}
        )
        report('warehouse-index creation', True)
    except Exception as e:
        report('warehouse-index creation', False, str(e))
        sys.exit(0)

# 6. Verify vector dimension
try:
    info = client.get_collection(collection_name)
    dim = info.config.params.vector_size
    report('Vector dimension', dim == 768, f'found {dim}')
except Exception as e:
    report('Vector dimension', False, str(e))

# 7. Verify retriever initialization (using app.retriever.QdrantRetriever)
try:
    from app.retriever import QdrantRetriever
    retriever = QdrantRetriever(collection_name)
    report('Retriever init', True)
except Exception as e:
    report('Retriever init', False, str(e))

print('Verification completed')
