import sys, os
sys.path.append(r'd:/rag/wm_rag')
from app.main import app
for route in app.routes:
    print(f"{route.path} -> {route.methods}")
