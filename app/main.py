import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .models import (
    WarehouseRequest, WarehouseResponse, WarehouseGatewayQuery, WarehouseGatewayResponse,
    IngestRequest
)
from .database import get_redis_client, get_mongodb_db, get_qdrant_client
from .rag_pipeline import RAGPipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import PointStruct
import uuid

load_dotenv()

logger = logging.getLogger(__name__)

# Configuration
SPRING_BOOT_URL = os.getenv("SPRING_BOOT_URL", "http://localhost:8081") # Gateway URL
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")

rag_pipeline = None
redis_client = None
mongo_db = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline, redis_client, mongo_db
    logger.info("Starting Warehouse AI service with Qdrant, MongoDB, Redis, and Gemini...")
    try:
        redis_client = get_redis_client(ping=False)
    except Exception as e:
        logger.warning(f"Failed to connect to Redis during startup: {e}. Will attempt lazy reconnection.")
    
    try:
        mongo_db = get_mongodb_db(ping=False)
    except Exception as e:
        logger.warning(f"Failed to connect to MongoDB during startup: {e}. Will attempt lazy reconnection.")

    try:
        rag_pipeline = RAGPipeline()
    except Exception as e:
        logger.warning(f"Failed to initialize RAGPipeline during startup: {e}. Will attempt lazy initialization.")
        
    logger.info("AI Service is ready.")
    yield
    logger.info("Shutting down...")

app = FastAPI(title="Warehouse Intelligent RAG Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions ---

async def save_query_to_gateway(query_data: WarehouseGatewayQuery):
    import httpx
    url = f"{SPRING_BOOT_URL}/api/ai/queries"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=query_data.model_dump(), timeout=5.0)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            logger.error(f"Gateway Query Error: {e}")
            return None

async def save_response_to_gateway(response_data: WarehouseGatewayResponse):
    import httpx
    url = f"{SPRING_BOOT_URL}/api/ai/responses"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=response_data.model_dump(), timeout=5.0)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            logger.error(f"Gateway Response Error: {e}")
            return None

# --- Main API Endpoints ---

@app.post("/api/ai/analyze", response_model=WarehouseResponse, summary="Analyze warehouse documents", description="Performs vector retrieval and LLM synthesis on warehouse domain questions.")
async def analyze_query(request: WarehouseRequest, background_tasks: BackgroundTasks):
    global rag_pipeline, redis_client, mongo_db
    if rag_pipeline is None:
        rag_pipeline = RAGPipeline()
    if redis_client is None:
        redis_client = get_redis_client()
    if mongo_db is None:
        mongo_db = get_mongodb_db()

    logger.info(f"Received analyze request: ocrDocumentId={request.ocrDocumentId} | attemptCount={request.attemptCount} | warehouseId={request.warehouseId}")

    # Use title as fallback if description is empty
    search_query = (request.description or "").strip() or request.title or ""
    desc_hash = hashlib.sha256(search_query.encode()).hexdigest()[:16]
    cache_key = f"warehouse_cache:{request.ocrDocumentId}:{request.warehouseId}:{desc_hash}:{request.attemptCount}"
    
    try:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            logger.info(f"[Redis] Cache hit for document {request.ocrDocumentId}")
            return WarehouseResponse(**json.loads(cached_data))
    except Exception as e:
        logger.warning(f"[Redis] Cache fetch error: {e}")

    # Audit Trail (optional field): Prioritize input, fallback to Redis
    recent_trail = request.auditTrail[-5:] if request.auditTrail else []
    audit_trail_text = "\n".join([
        f"- {a.auditDate or ''}: {a.actionTaken or ''} — {a.resolution or a.description or ''}"
        for a in recent_trail
    ])

    if request.ocrDocumentId:
        if audit_trail_text:
            try:
                redis_client.setex(f"audit_trail:{request.ocrDocumentId}", 86400, audit_trail_text)
            except Exception as e:
                logger.debug(f"[Redis] Audit trail store error: {e}")
        else:
            try:
                cached_trail = redis_client.get(f"audit_trail:{request.ocrDocumentId}")
                if cached_trail:
                    audit_trail_text = cached_trail if isinstance(cached_trail, str) else cached_trail.decode('utf-8')
            except Exception as e:
                logger.debug(f"[Redis] Audit trail fetch error: {e}")

    # Retrieve previous response from Redis if follow-up mode
    prev_response = None
    if request.attemptCount > 1 and request.ocrDocumentId:
        try:
            stored = redis_client.get(f"prev_response:{request.ocrDocumentId}")
            if stored:
                try:
                    stored_str = stored if isinstance(stored, str) else stored.decode('utf-8')
                    prev_list = json.loads(stored_str)
                    prev_response = "\n---\n".join(prev_list)
                except (json.JSONDecodeError, ValueError):
                    prev_response = stored if isinstance(stored, str) else stored.decode('utf-8')
        except Exception as e:
            logger.debug(f"[Redis] Previous response fetch error: {e}")
    if not prev_response:
        prev_response = request.previousResponse

    # Execute modular RAG Flow
    suggestion, final_confidence, predicted_category, top_ocr_doc_id = await rag_pipeline.execute(
        query=search_query,
        document_type=request.documentType,
        audit_trail_text=audit_trail_text,
        attempt_count=request.attemptCount,
        prev_response=prev_response,
        follow_up_question=request.followUpQuestion,
        priority=request.priority
    )

    # Persistence to local MongoDB and outer Gateway (via BackgroundTasks for safety)
    async def persist():
        try:
            # 1. MongoDB Local logging
            mongo_db.queries.insert_one({
                "userId": request.userId,
                "warehouseId": request.warehouseId,
                "documentType": request.documentType or "Unknown",
                "question": search_query,
                "timestamp": datetime.now(timezone.utc)
            })
            mongo_db.responses.insert_one({
                "userId": request.userId,
                "warehouseId": request.warehouseId,
                "ocrDocumentId": f"DOC-{request.ocrDocumentId}" if request.ocrDocumentId else top_ocr_doc_id,
                "answer": suggestion,
                "confidence": final_confidence,
                "status": "ANALYZED",
                "title": request.title or "General",
                "description": search_query,
                "timestamp": datetime.now(timezone.utc)
            })
            logger.info("[MongoDB] Successfully logged query and response")
        except Exception as e:
            logger.error(f"[MongoDB] Persistence error: {e}")

        # 2. External Gateway (using updated WarehouseGatewayQuery and WarehouseGatewayResponse)
        query = await save_query_to_gateway(
            WarehouseGatewayQuery(
                userId=request.userId,
                warehouseId=request.warehouseId or "Unknown",
                documentType=request.documentType or "Unknown",
                question=search_query
            )
        )
        if query and "id" in query:
            await save_response_to_gateway(
                WarehouseGatewayResponse(
                    queryId=query["id"],
                    userId=request.userId,
                    warehouseId=request.warehouseId or "Unknown",
                    ocrDocumentId=f"DOC-{request.ocrDocumentId}" if request.ocrDocumentId else top_ocr_doc_id,
                    answer=suggestion,
                    confidence=final_confidence,
                    status="ANALYZED",
                    title=request.title or "General",
                    description=search_query
                )
            )

    background_tasks.add_task(persist)

    res_obj = WarehouseResponse(
        suggestion=suggestion,
        confidence=final_confidence,
        predictedCategory=predicted_category,
        status="SUCCESS"
    )

    # Store interaction in Redis for future follow-ups
    if request.ocrDocumentId:
        try:
            stored = redis_client.get(f"prev_response:{request.ocrDocumentId}")
            prev_list = []
            if stored:
                try:
                    stored_str = stored if isinstance(stored, str) else stored.decode('utf-8')
                    prev_list = json.loads(stored_str)
                except (json.JSONDecodeError, ValueError):
                    prev_list = [stored if isinstance(stored, str) else stored.decode('utf-8')]
            
            entry = f"AI: {suggestion}" if request.attemptCount <= 1 else f"User: {request.followUpQuestion}\nAI: {suggestion}"
            prev_list.append(entry)
            
            # Keep only the last 2 interactions
            redis_client.setex(f"prev_response:{request.ocrDocumentId}", 86400, json.dumps(prev_list[-2:]))
            logger.debug(f"[Redis] Stored prev_response key: prev_response:{request.ocrDocumentId}")
        except Exception as e:
            logger.warning(f"[Redis] prev_response store error: {e}")

    # Cache response only if it's a valid complete answer (not insufficient/error)
    if "INSUFFICIENT" not in suggestion and "unavailable" not in suggestion:
        try:
            redis_client.setex(cache_key, 3600, res_obj.model_dump_json())
            logger.debug(f"[Redis] Cached response key: {cache_key}")
        except Exception as e:
            logger.warning(f"[Redis] Cache store error: {e}")

    return res_obj

@app.post("/api/rag/ingest", response_model=dict, summary="Ingest OCR document text", description="Chunk OCR text, embed, and store in Qdrant warehouse-index.")
async def rag_ingest(request: IngestRequest):
    # Initialize components
    global rag_pipeline
    if rag_pipeline is None:
        rag_pipeline = RAGPipeline()
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-3.5-turbo",
        chunk_size=500,
        chunk_overlap=50,
    )
    # Split raw text into chunks
    chunk_texts = splitter.split_text(request.text)
    points = []
    for idx, chunk_text in enumerate(chunk_texts):
        embedding = rag_pipeline.core_pipeline.embedder.embed(chunk_text)
        # deterministic UUID per chunk
        point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{request.ocr_document_id}_{idx}"))
        metadata = {
            "ocr_document_id": request.ocr_document_id,
            "document_type": request.document_type,
            "warehouse_id": request.warehouse_id,
            "chunk_index": idx,
            "text": chunk_text,
        }
        # Use same payload structure as core pipeline for consistency
        points.append(PointStruct(id=point_uuid, vector=embedding, payload={"metadata": metadata, **metadata}))
        # batch upsert every 50 points
        if len(points) >= 50:
            rag_pipeline.core_pipeline.vector_store._client.upsert(collection_name="warehouse-index", points=points)
            points = []
    # upsert any remaining points
    if points:
        rag_pipeline.core_pipeline.vector_store._client.upsert(collection_name="warehouse-index", points=points)
    return {"status": "SUCCESS", "chunks_created": len(chunk_texts)}

@app.get("/status", summary="Get service status", description="Returns connection and activation status of the RAG service components.")
async def get_status():
    return {"status": "active", "integration": "Warehouse RAG Service"}
