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
    IngestRequest, DocumentListItem, DocumentDetailResponse, CollectionStatsResponse, ChunkDetail
)
from .database import get_redis_client, get_db_session, get_qdrant_client
from .db_models import OCRDocumentModel
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline, redis_client
    logger.info("Starting Warehouse AI service with Qdrant, PostgreSQL, Redis, and Gemini...")
    # Redis is disabled to prevent connection hangs
    redis_client = None

    # Lazy initialize RAGPipeline on first request to avoid PyTorch deadlock under ASGI/Uvicorn lifespan
    logger.info("RAGPipeline will be initialized lazily on the first request to prevent event loop deadlocks.")
        
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
    global rag_pipeline, redis_client
    if rag_pipeline is None:
        rag_pipeline = RAGPipeline()
    
    # Redis is disabled to prevent connection hangs
    redis_client = None

    logger.info(f"Received analyze request: ocrDocumentId={request.ocrDocumentId} | attemptCount={request.attemptCount} | warehouseId={request.warehouseId}")

    # Use title as fallback if description is empty
    search_query = (request.description or "").strip() or request.title or ""
    desc_hash = hashlib.sha256(search_query.encode()).hexdigest()[:16]
    cache_key = f"warehouse_cache:{request.ocrDocumentId}:{request.warehouseId}:{desc_hash}:{request.attemptCount}"
    
    try:
        cached_data = redis_client.get(cache_key) if redis_client else None
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
                if redis_client:
                    redis_client.setex(f"audit_trail:{request.ocrDocumentId}", 86400, audit_trail_text)
            except Exception as e:
                logger.debug(f"[Redis] Audit trail store error: {e}")
        else:
            try:
                cached_trail = redis_client.get(f"audit_trail:{request.ocrDocumentId}") if redis_client else None
                if cached_trail:
                    audit_trail_text = cached_trail if isinstance(cached_trail, str) else cached_trail.decode('utf-8')
            except Exception as e:
                logger.debug(f"[Redis] Audit trail fetch error: {e}")

    # Retrieve previous response from Redis if follow-up mode
    prev_response = None
    if request.attemptCount > 1 and request.ocrDocumentId:
        try:
            stored = redis_client.get(f"prev_response:{request.ocrDocumentId}") if redis_client else None
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

    # Look up WMS metadata filters in PostgreSQL if ocrDocumentId is provided
    if request.ocrDocumentId:
        try:
            db = get_db_session()
            try:
                doc = db.query(OCRDocumentModel).filter(OCRDocumentModel.id == request.ocrDocumentId).first()
                if doc:
                    doc_type = doc_type or doc.document_type
                    wh_id = wh_id or doc.warehouse_id
                    request.sku = request.sku or doc.sku
                    prod_id = prod_id or doc.product_id
                    request.category = request.category or doc.category
                    request.zone = request.zone or doc.zone
                    request.rack = request.rack or doc.rack
                    request.shelf = request.shelf or doc.shelf
                    request.bin = request.bin or doc.bin
            finally:
                db.close()
        except Exception as db_err:
            logger.warning(f"[Database] Failed to look up metadata for {request.ocrDocumentId}: {db_err}")

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

    # Persistence to outer Gateway (via BackgroundTasks for safety)
    async def persist():
        # External Gateway (using updated WarehouseGatewayQuery and WarehouseGatewayResponse)
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
            stored = redis_client.get(f"prev_response:{request.ocrDocumentId}") if redis_client else None
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
            if redis_client:
                redis_client.setex(f"prev_response:{request.ocrDocumentId}", 86400, json.dumps(prev_list[-2:]))
            logger.debug(f"[Redis] Stored prev_response key: prev_response:{request.ocrDocumentId}")
        except Exception as e:
            logger.warning(f"[Redis] prev_response store error: {e}")

    # Cache response only if it's a valid complete answer (not insufficient/error)
    if "INSUFFICIENT" not in suggestion and "unavailable" not in suggestion:
        try:
            if redis_client:
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

        # Use same payload structure as core pipeline for consistency
        points.append(PointStruct(id=point_uuid, vector=embedding, payload={"metadata": metadata, **metadata}))
        # batch upsert every 50 points
        if len(points) >= 50:
            rag_pipeline.core_pipeline.vector_store._client.upsert(collection_name="warehouse-index", points=points)
            points = []
    # upsert any remaining points
    if points:
        rag_pipeline.core_pipeline.vector_store._client.upsert(collection_name="warehouse-index", points=points)

    # Upsert into PostgreSQL for local metadata tracking
    try:
        db = get_db_session()
        try:
            doc = db.query(OCRDocumentModel).filter(OCRDocumentModel.id == request.ocr_document_id).first()
            now = datetime.now(timezone.utc)
            if not doc:
                # If not exists, insert a new row to support flexible testing/dynamic indexing
                doc = OCRDocumentModel(
                    id=request.ocr_document_id,
                    file_name=f"Ingested Document {request.ocr_document_id}",
                    file_path="",
                    document_type=request.document_type,
                    raw_text=request.text,
                    processing_status="COMPLETED",
                    created_at=now,
                )
                db.add(doc)
            
            # Update all metadata fields and chunk count
            doc.document_type = request.document_type
            doc.warehouse_id = request.warehouse_id
            doc.sku = request.sku
            doc.product_id = request.product_id
            doc.category = request.category
            doc.zone = request.zone
            doc.rack = request.rack
            doc.shelf = request.shelf
            doc.bin = request.bin
            doc.chunk_count = len(chunk_texts)
            doc.updated_at = now
            doc.processing_status = "COMPLETED"
            
            db.commit()
            logger.info(f"[Database] Successfully logged metadata in PostgreSQL for {request.ocr_document_id}")
        except Exception as db_err:
            db.rollback()
            logger.error(f"[Database] Failed to upsert metadata: {db_err}")
        finally:
            db.close()
    except Exception as db_init_err:
        logger.error(f"[Database] Connection error on ingest metadata logging: {db_init_err}")

    return {"status": "SUCCESS", "chunks_created": len(chunk_texts)}

@app.get("/status", summary="Get service status", description="Returns connection and activation status of the RAG service components.")
async def get_status():
    postgres_status = "Connected"
    try:
        from sqlalchemy import text
        db = get_db_session()
        try:
            db.execute(text("SELECT 1"))
        finally:
            db.close()
    except Exception as e:
        postgres_status = f"Failed: {str(e)}"

    qdrant_status = "Connected"
    try:
        qdrant_client = get_qdrant_client()
        qdrant_client.get_collection(collection_name="warehouse-index")
    except Exception as e:
        qdrant_status = f"Failed: {str(e)}"

    gemini_status = "Connected"
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        gemini_status = "Failed: GEMINI_API_KEY environment variable is not set"

    # Redis check disabled
    redis_status = "Disabled (Optional)"

    is_healthy = (
        postgres_status == "Connected"
        and qdrant_status == "Connected"
        and gemini_status == "Connected"
    )
    service_status = "active" if is_healthy else "degraded"

    return {
        "status": service_status,
        "postgresql": postgres_status,
        "qdrant": qdrant_status,
        "gemini": gemini_status,
        "redis": redis_status,
        "integration": "Warehouse RAG Service"
    }

@app.get("/api/rag/documents", response_model=List[DocumentListItem], summary="List all indexed documents")
async def list_documents():
    try:
        db = get_db_session()
        try:
            # Query documents that have been ingested (chunk_count > 0)
            results = db.query(OCRDocumentModel).filter(OCRDocumentModel.chunk_count > 0).all()
            documents = []
            for item in results:
                documents.append(
                    DocumentListItem(
                        ocr_document_id=item.id,
                        document_type=item.document_type,
                        warehouse_id=item.warehouse_id,
                        sku=item.sku,
                        product_id=item.product_id,
                        category=item.category,
                        zone=item.zone,
                        rack=item.rack,
                        shelf=item.shelf,
                        bin=item.bin,
                        chunk_count=item.chunk_count,
                        updated_at=item.updated_at or item.created_at
                    )
                )
            return documents
        finally:
            db.close()
    except Exception as e:
        logger.error(f"[Database] Failed to list documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database aggregation error: {str(e)}"
        )

@app.get("/api/rag/documents/{document_id}", response_model=DocumentDetailResponse, summary="Get document details and chunks")
async def get_document_details(document_id: str):
    # 1. Fetch document metadata from PostgreSQL
    doc = None
    try:
        db = get_db_session()
        try:
            doc = db.query(OCRDocumentModel).filter(OCRDocumentModel.id == document_id).first()
        finally:
            db.close()
    except Exception as e:
        logger.error(f"[Database] Failed to query document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query error: {str(e)}"
        )

    if not doc or doc.chunk_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found."
        )

    # 2. Retrieve chunk texts dynamically from Qdrant vectors payload scroll
    chunk_list = []
    try:
        qdrant_client = get_qdrant_client()
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        points, _ = qdrant_client.scroll(
            collection_name="warehouse-index",
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.ocr_document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            ),
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        
        for p in points:
            meta = p.payload.get("metadata", {})
            chunk_list.append(
                ChunkDetail(
                    chunk_id=meta.get("chunk_id") or str(p.id),
                    chunk_index=meta.get("chunk_index", 0),
                    text=meta.get("text", "")
                )
            )
        # Sort chunks by index
        chunk_list.sort(key=lambda x: x.chunk_index)
        
    except Exception as q_err:
        logger.error(f"[Qdrant] Failed to fetch chunks for document {document_id}: {q_err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector database retrieval error: {str(q_err)}"
        )

    if not chunk_list:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document chunks for ID {document_id} not found in vector store."
        )

    return DocumentDetailResponse(
        ocr_document_id=document_id,
        document_type=doc.document_type,
        warehouse_id=doc.warehouse_id,
        sku=doc.sku,
        product_id=doc.product_id,
        category=doc.category,
        zone=doc.zone,
        rack=doc.rack,
        shelf=doc.shelf,
        bin=doc.bin,
        chunks=chunk_list
    )

@app.delete("/api/rag/documents/{document_id}", summary="Delete document vectors and metadata")
async def delete_document(document_id: str):
    # 1. Delete vectors from Qdrant
    qdrant_deleted_count = 0
    try:
        qdrant_client = get_qdrant_client()
        from qdrant_client.models import FilterSelector, Filter, FieldCondition, MatchValue
        
        points, _ = qdrant_client.scroll(
            collection_name="warehouse-index",
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.ocr_document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            ),
            limit=1000,
            with_payload=False,
            with_vectors=False
        )
        qdrant_deleted_count = len(points)
        
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

    # 2. Reset related PostgreSQL metadata fields instead of deleting the row
    # (or delete the row if it's a dynamic mock document with a simple string ID that isn't a valid UUID)
    deleted_count = qdrant_deleted_count
    try:
        db = get_db_session()
        try:
            doc = db.query(OCRDocumentModel).filter(OCRDocumentModel.id == document_id).first()
            if doc:
                if doc.chunk_count > 0:
                    deleted_count = doc.chunk_count
                
                is_uuid = False
                try:
                    uuid.UUID(document_id)
                    is_uuid = True
                except ValueError:
                    pass

                if is_uuid:
                    doc.chunk_count = 0
                    doc.warehouse_id = None
                    doc.sku = None
                    doc.product_id = None
                    doc.category = None
                    doc.zone = None
                    doc.rack = None
                    doc.shelf = None
                    doc.bin = None
                    doc.processing_status = "UPLOADED"
                else:
                    # Dynamically generated mock/test doc: safe to hard delete
                    db.delete(doc)
                db.commit()
                logger.info(f"[Database] Reset RAG metadata in PostgreSQL for {document_id}")
        except Exception as db_err:
            db.rollback()
            logger.error(f"[Database] Failed to clear metadata for {document_id}: {db_err}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database deletion error: {str(db_err)}"
            )
        finally:
            db.close()
    except Exception as db_init_err:
        logger.error(f"[Database] Connection error on delete metadata: {db_init_err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database connection error: {str(db_init_err)}"
        )

    return {"status": "SUCCESS", "deleted_count": deleted_count}

@app.get("/api/rag/collections/stats", response_model=CollectionStatsResponse, summary="Get database and vector statistics")
async def get_collections_stats():
    # 1. Total unique documents & chunks in PostgreSQL
    total_documents = 0
    total_chunks = 0
    try:
        db = get_db_session()
        try:
            results = db.query(OCRDocumentModel).filter(OCRDocumentModel.chunk_count > 0).all()
            total_documents = len(results)
            from sqlalchemy import func
            total_chunks = db.query(func.sum(OCRDocumentModel.chunk_count)).filter(OCRDocumentModel.chunk_count > 0).scalar() or 0
        finally:
            db.close()
    except Exception as e:
        logger.error(f"[Database] Failed to count stats: {e}")

    # 2. Total points in Qdrant warehouse-index
    vector_count = 0
    try:
        qdrant_client = get_qdrant_client()
        collection_info = qdrant_client.get_collection(collection_name="warehouse-index")
        vector_count = collection_info.points_count
    except Exception as q_err:
        logger.warning(f"[Qdrant] Failed to fetch collection info: {q_err}")

    return CollectionStatsResponse(
        collection_name="warehouse-index",
        total_documents=total_documents,
        total_chunks=total_chunks,
        vector_count=vector_count
    )
