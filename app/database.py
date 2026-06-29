import os
import logging
import redis
from qdrant_client import QdrantClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
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

# Database Configuration (SQLAlchemy)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///../wm_backend/db.sqlite3")
if DATABASE_URL.startswith("sqlite:///"):
    db_relative_path = DATABASE_URL.replace("sqlite:///", "")
    # Handle absolute resolution on Windows
    db_absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", db_relative_path))
    DATABASE_URL = f"sqlite:///{db_absolute_path.replace('\\', '/')}"

# Lazy initialization of clients to prevent overhead on import
_redis_client = None
_qdrant_client = None
_engine = None
_SessionFactory = None

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
            timeout=30.0,
        )
    return _qdrant_client

def get_db_session():
    global _engine, _SessionFactory
    if _SessionFactory is None:
        logger.info(f"[Database] Initializing SQLAlchemy engine for database: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL}")
        connect_args = {}
        if DATABASE_URL.startswith("sqlite:"):
            connect_args = {"check_same_thread": False}
        
        _engine = create_engine(
            DATABASE_URL,
            connect_args=connect_args,
            pool_pre_ping=True
        )
        _SessionFactory = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
        
        # Proactively create tables if they do not exist (useful for testing environments)
        try:
            from .db_models import Base
            Base.metadata.create_all(_engine)
        except Exception as e:
            logger.warning(f"[Database] Tables auto-creation skipped or failed: {e}")
            
    return _SessionFactory()

