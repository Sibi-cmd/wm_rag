import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

_embedding_model = None

def get_embeddings_model():
    global _embedding_model
    if _embedding_model is None:
        model_name = os.getenv("EMBEDDINGS_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")
        print(f"Loading SentenceTransformer model ({model_name})...", flush=True)
        _embedding_model = SentenceTransformer(model_name)
        print("Model loaded.", flush=True)
    return _embedding_model

def get_embedding(text: str):
    """Generate embedding vector for a given text input."""
    model = get_embeddings_model()
    return model.encode(text).tolist()
