import sys, os
sys.path.append('d:/rag/wm_rag')

from app.database import get_qdrant_client

client = get_qdrant_client()
col_name = 'warehouse-index'
info = client.get_collection(collection_name=col_name)
print('Collection info:')
print(info)
# Payload indexes
if hasattr(info, 'payload_schema'):
    print('Payload indexes:')
    for field, schema in info.payload_schema.items():
        print(f' - {field}: {schema}')
else:
    print('No payload_schema attribute')
