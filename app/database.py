import os
import logging
import redis
from qdrant_client import QdrantClient
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Configure logger for this module
logger = logging.getLogger(__name__)

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "wm_rag")

# Lazy initialization of clients to prevent overhead on import
_redis_client = None
_qdrant_client = None
_mongo_client = None

def get_redis_client(ping=True):
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            decode_responses=True,
            socket_timeout=2.0,
            socket_connect_timeout=2.0,
        )
        if ping:
            try:
                _redis_client.ping()
                logger.info("[Redis] Connected to Redis successfully")
            except Exception as e:
                logger.error(f"[Redis] Connection error: {e}")
                raise
    return _redis_client

def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=2.0,
        )
    return _qdrant_client

def get_mongodb_db(ping=True):
    global _mongo_client
    if _mongo_client is None:
        # serverSelectionTimeoutMS ensures we don't hang indefinitely on connection attempts
        _mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=2000)
        if ping:
            try:
                _mongo_client.server_info()
                logger.info("[MongoDB] Connected to MongoDB successfully")
            except Exception as e:
                logger.error(f"[MongoDB] Connection error: {e}")
                raise
    return _mongo_client[MONGODB_DB_NAME]

# Backwards‑compatible global client (may be used elsewhere)
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=True,
    socket_timeout=2.0,
    socket_connect_timeout=2.0,
)
