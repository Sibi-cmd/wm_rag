import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .models import (
    WarehouseRequest, WarehouseResponse, WarehouseGatewayQuery, WarehouseGatewayResponse,
    IngestRequest, DocumentListItem, DocumentDetailResponse, CollectionStatsResponse,
    ChunkDetail, DocumentDeleteResponse, DocumentMetadata
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

    # Extract filters with camelCase / snake_case backward compatibility
    doc_type = request.document_type or request.documentType
    prod_id = request.product_id or request.productId
    wh_id = request.warehouse_id or request.warehouseId

    # Execute modular RAG Flow
    suggestion, final_confidence, predicted_category, top_ocr_doc_id = await rag_pipeline.execute(
        query=search_query,
        document_type=doc_type,
        sku=request.sku,
        product_id=prod_id,
        category=request.category,
        warehouse_id=wh_id,
        zone=request.zone,
        rack=request.rack,
        shelf=request.shelf,
        bin=request.bin,
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
    global rag_pipeline, mongo_db
    if rag_pipeline is None:
        rag_pipeline = RAGPipeline()
    if mongo_db is None:
        try:
            mongo_db = get_mongodb_db()
        except Exception as e:
            logger.warning(f"Failed to connect to MongoDB during ingestion: {e}")

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
            "sku": request.sku,
            "product_id": request.product_id,
            "category": request.category,
            "zone": request.zone,
            "rack": request.rack,
            "shelf": request.shelf,
            "bin": request.bin,
            "chunk_index": idx,
            "text": chunk_text,
        }

        # Upsert into MongoDB for local chunk tracking
        if mongo_db is not None:
            try:
                mongo_db.chunks.update_one(
                    {"chunk_id": f"{request.ocr_document_id}_{idx}"},
                    {"$set": {
                        "chunk_id": f"{request.ocr_document_id}_{idx}",
                        "ocr_document_id": request.ocr_document_id,
                        "document_type": request.document_type,
                        "warehouse_id": request.warehouse_id,
                        "sku": request.sku,
                        "product_id": request.product_id,
                        "category": request.category,
                        "zone": request.zone,
                        "rack": request.rack,
                        "shelf": request.shelf,
                        "bin": request.bin,
                        "chunk_index": idx,
                        "text": chunk_text,
                        "updated_at": datetime.now(timezone.utc)
                    }},
                    upsert=True
                )
            except Exception as mongo_err:
                logger.warning(f"[MongoDB] Chunk log error: {mongo_err}")

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

@app.get("/api/rag/documents", response_model=List[DocumentListItem], summary="List all indexed documents")
async def list_documents():
    global mongo_db
    results = []
    mongo_success = False

    try:
        if mongo_db is None:
            mongo_db = get_mongodb_db(ping=False)
        pipeline = [
            {
                "$group": {
                    "_id": "$ocr_document_id",
                    "chunk_count": {"$sum": 1},
                    "document_type": {"$first": "$document_type"},
                    "warehouse_id": {"$first": "$warehouse_id"},
                    "sku": {"$first": "$sku"},
                    "zone": {"$first": "$zone"},
                    "rack": {"$first": "$rack"},
                    "shelf": {"$first": "$shelf"},
                    "bin": {"$first": "$bin"}
                }
            }
        ]
        results = list(mongo_db.chunks.aggregate(pipeline))
        mongo_success = True
    except Exception as e:
        logger.warning(f"[MongoDB] Failed to aggregate documents, falling back to Qdrant: {e}")

    documents = []

    if mongo_success and results:
        for item in results:
            documents.append(
                DocumentListItem(
                    document_id=item["_id"],
                    document_type=item.get("document_type"),
                    warehouse_id=item.get("warehouse_id"),
                    sku=item.get("sku"),
                    zone=item.get("zone"),
                    rack=item.get("rack"),
                    shelf=item.get("shelf"),
                    bin=item.get("bin"),
                    chunk_count=item["chunk_count"]
                )
            )
    else:
        # Fallback to Qdrant Scroll Strategy
        try:
            qdrant_client = get_qdrant_client()
            offset = None
            doc_map = {}
            while True:
                scroll_res = qdrant_client.scroll(
                    collection_name="warehouse-index",
                    limit=100,
                    with_payload=True,
                    with_vectors=False,
                    offset=offset
                )
                records, offset = scroll_res
                for record in records:
                    payload = record.payload or {}
                    meta = payload.get("metadata", payload)
                    doc_id = meta.get("ocr_document_id") or meta.get("document_id")
                    if not doc_id:
                        continue
                    if doc_id not in doc_map:
                        doc_map[doc_id] = {
                            "document_id": doc_id,
                            "document_type": meta.get("document_type"),
                            "warehouse_id": meta.get("warehouse_id"),
                            "sku": meta.get("sku"),
                            "zone": meta.get("zone"),
                            "rack": meta.get("rack"),
                            "shelf": meta.get("shelf"),
                            "bin": meta.get("bin"),
                            "chunk_count": 0
                        }
                    doc_map[doc_id]["chunk_count"] += 1
                if offset is None:
                    break
            
            for doc in doc_map.values():
                documents.append(DocumentListItem(**doc))
        except Exception as q_err:
            logger.error(f"[Qdrant] Failed to scroll collection: {q_err}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve documents from storage: {str(q_err)}"
            )

    return documents

@app.get("/api/rag/documents/{document_id}", response_model=DocumentDetailResponse, summary="Get document details and chunks")
async def get_document_details(document_id: str):
    global mongo_db
    chunks = []
    mongo_success = False

    try:
        if mongo_db is None:
            mongo_db = get_mongodb_db(ping=False)
        chunks_cursor = mongo_db.chunks.find({"ocr_document_id": document_id}).sort("chunk_index", 1)
        chunks = list(chunks_cursor)
        mongo_success = True
    except Exception as e:
        logger.warning(f"[MongoDB] Failed to retrieve document details, falling back to Qdrant: {e}")

    if not mongo_success or not chunks:
        # Fallback to Qdrant Scroll with filter
        try:
            qdrant_client = get_qdrant_client()
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            filter_cond = Filter(
                must=[
                    FieldCondition(
                        key="metadata.ocr_document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
            
            offset = None
            q_chunks = []
            while True:
                scroll_res = qdrant_client.scroll(
                    collection_name="warehouse-index",
                    scroll_filter=filter_cond,
                    limit=100,
                    with_payload=True,
                    with_vectors=False,
                    offset=offset
                )
                records, offset = scroll_res
                for record in records:
                    payload = record.payload or {}
                    meta = payload.get("metadata", payload)
                    q_chunks.append(meta)
                if offset is None:
                    break
            
            if not q_chunks:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document with ID {document_id} not found."
                )
            
            # Sort the Qdrant records by chunk_index
            q_chunks.sort(key=lambda x: int(x.get("chunk_index", 0)))
            
            first_chunk = q_chunks[0]
            chunk_list = [
                ChunkDetail(
                    chunk_index=int(c.get("chunk_index", 0)),
                    text=c.get("text", "")
                )
                for c in q_chunks
            ]
            
            return DocumentDetailResponse(
                metadata=DocumentMetadata(
                    document_id=document_id,
                    document_type=first_chunk.get("document_type"),
                    warehouse_id=first_chunk.get("warehouse_id")
                ),
                chunks=chunk_list
            )
        except HTTPException:
            raise
        except Exception as q_err:
            logger.error(f"[Qdrant] Scroll error for document {document_id}: {q_err}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve document details: {str(q_err)}"
            )

    first_chunk = chunks[0]
    chunk_list = [
        ChunkDetail(
            chunk_index=c.get("chunk_index", 0),
            text=c.get("text", "")
        )
        for c in chunks
    ]

    return DocumentDetailResponse(
        metadata=DocumentMetadata(
            document_id=document_id,
            document_type=first_chunk.get("document_type"),
            warehouse_id=first_chunk.get("warehouse_id")
        ),
        chunks=chunk_list
    )

@app.delete("/api/rag/documents/{document_id}", response_model=DocumentDeleteResponse, summary="Delete document vectors and metadata")
async def delete_document(document_id: str):
    global mongo_db
    if mongo_db is None:
        mongo_db = get_mongodb_db(ping=False)

    # 1. Delete vectors from Qdrant
    try:
        qdrant_client = get_qdrant_client()
        from qdrant_client.models import FilterSelector, Filter, FieldCondition, MatchValue
        qdrant_client.delete(
            collection_name="warehouse-index",
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.ocr_document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            )
        )
    except Exception as q_err:
        logger.warning(f"[Qdrant] Failed to delete points for document {document_id}: {q_err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete points from vector store: {str(q_err)}"
        )

    # 2. Delete related MongoDB records
    try:
        delete_result = mongo_db.chunks.delete_many({"ocr_document_id": document_id})
        deleted_count = delete_result.deleted_count
    except Exception as m_err:
        logger.error(f"[MongoDB] Failed to delete chunks for {document_id}: {m_err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database deletion error: {str(m_err)}"
        )

    return DocumentDeleteResponse(
        status="SUCCESS",
        document_id=document_id
    )

@app.get("/api/rag/collections/stats", response_model=CollectionStatsResponse, summary="Get database and vector statistics")
async def get_collections_stats():
    global mongo_db
    
    # 1. Total points in Qdrant warehouse-index
    vector_count = 0
    collection_name = "warehouse-index"
    try:
        qdrant_client = get_qdrant_client()
        collection_info = qdrant_client.get_collection(collection_name=collection_name)
        vector_count = collection_info.points_count
    except Exception as q_err:
        logger.warning(f"[Qdrant] Failed to fetch collection info: {q_err}")

    # 2. Total unique documents & chunks in MongoDB
    total_documents = 0
    total_chunks = 0
    mongo_success = False
    try:
        if mongo_db is None:
            mongo_db = get_mongodb_db(ping=False)
        total_documents = len(mongo_db.chunks.distinct("ocr_document_id"))
        total_chunks = mongo_db.chunks.count_documents({})
        mongo_success = True
    except Exception as e:
        logger.warning(f"[MongoDB] Failed to count stats: {e}")

    # Fallback to estimating stats from Qdrant if MongoDB is down or empty, but Qdrant has vectors
    if (not mongo_success or total_chunks == 0) and vector_count > 0:
        try:
            qdrant_client = get_qdrant_client()
            offset = None
            unique_docs = set()
            while True:
                scroll_res = qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=100,
                    with_payload=True,
                    with_vectors=False,
                    offset=offset
                )
                records, offset = scroll_res
                for record in records:
                    payload = record.payload or {}
                    meta = payload.get("metadata", payload)
                    doc_id = meta.get("ocr_document_id") or meta.get("document_id")
                    if doc_id:
                        unique_docs.add(doc_id)
                if offset is None:
                    break
            total_documents = len(unique_docs)
            total_chunks = vector_count
        except Exception as q_err:
            logger.warning(f"[Qdrant] Scroll estimation failed: {q_err}")

    return CollectionStatsResponse(
        collection_name=collection_name,
        total_documents=total_documents,
        total_chunks=total_chunks or vector_count,
        vector_count=vector_count
    )
