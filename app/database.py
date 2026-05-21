import os
import psycopg2
import redis
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

def get_db_connection():
    conn = psycopg2.connect(
        DATABASE_URL, 
        cursor_factory=RealDictCursor, 
        connect_timeout=10,
        keepalives=1, 
        keepalives_idle=30, 
        keepalives_interval=10, 
        keepalives_count=5
    )
    return conn

# Global Redis client instance
redis_client = redis.Redis(
    host=REDIS_HOST, 
    port=REDIS_PORT, 
    password=REDIS_PASSWORD,
    decode_responses=True
)

def get_redis_client():
    return redis_client
