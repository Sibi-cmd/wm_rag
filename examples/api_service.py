"""
examples.api_service
~~~~~~~~~~~~~~~~~~~~
Helper script to start the standalone FastAPI RAG Middleman Service.
"""

import sys
from pathlib import Path

# Add project root to python path so rag_core is resolved successfully
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from dotenv import load_dotenv

load_dotenv()


def main():
    print("=====================================================================")
    print("             *** Starting RAG Middleman Service API Server ***       ")
    print("=====================================================================")
    print(" * Dashboard URL: http://localhost:8000/")
    print(" * Interactive OpenAPI Docs: http://localhost:8000/docs")
    print("=====================================================================")
    
    # Run uvicorn server
    uvicorn.run(
        "rag_core.api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
