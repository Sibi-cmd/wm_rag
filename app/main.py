from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import json
import hashlib
import asyncio
from datetime import datetime
from dotenv import load_dotenv

from .models import (
    AIRequest, AIResponse, GatewayQueryCreate, GatewayResponseCreate
)
from .database import get_redis_client, get_mongodb_db
from .rag_pipeline import RAGPipeline

load_dotenv()

# Configuration
SPRING_BOOT_URL = os.getenv("SPRING_BOOT_URL", "http://localhost:8081") # Gateway URL

rag_pipeline = None
redis_client = None
mongo_db = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline, redis_client, mongo_db
    print("Starting AI service with Qdrant, MongoDB, Redis, and Gemini...", flush=True)
    redis_client = get_redis_client()
    mongo_db = get_mongodb_db()
    rag_pipeline = RAGPipeline()
    print("AI Service is ready.", flush=True)
    yield
    print("Shutting down...", flush=True)

app = FastAPI(title="EV Fleet AI Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions ---

async def save_query_to_gateway(query_data: GatewayQueryCreate):
    import httpx
    url = f"{SPRING_BOOT_URL}/api/ai/queries"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=query_data.dict(), timeout=5.0)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            print(f"Gateway Query Error: {e}", flush=True)
            return None

async def save_response_to_gateway(response_data: GatewayResponseCreate):
    import httpx
    url = f"{SPRING_BOOT_URL}/api/ai/responses"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=response_data.dict(), timeout=5.0)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            print(f"Gateway Response Error: {e}", flush=True)
            return None

# --- Main API Endpoints ---

@app.post("/api/ai/analyze", response_model=AIResponse)
async def analyze_complaint(request: AIRequest):
    global rag_pipeline, redis_client, mongo_db
    if rag_pipeline is None:
        rag_pipeline = RAGPipeline()
    if redis_client is None:
        redis_client = get_redis_client()
    if mongo_db is None:
        mongo_db = get_mongodb_db()

    # --- DEBUG: Log full received payload ---
    print("=" * 60)
    print("RECEIVED PAYLOAD:", request.dict())
    print(f"complaintId={request.complaintId} | attemptCount={request.aiAttemptCount} | vehicleId={request.vehicleId}")
    print("=" * 60)

    # Use title as fallback if description is empty (Spring Boot may send empty string/null)
    search_query = (request.description or "").strip() or request.title or ""
    desc_hash = hashlib.md5(search_query.encode()).hexdigest()
    cache_key = f"ai_cache:{request.complaintId}:{request.vehicleId}:{desc_hash}:{request.aiAttemptCount}"
    
    try:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            print(f"[Redis] Cache hit for complaint {request.complaintId}", flush=True)
            return AIResponse(**json.loads(cached_data))
    except Exception as e:
        print(f"[Redis] Cache fetch error: {e}", flush=True)

    # Service History: Prioritize Spring Boot input, fallback to Redis
    recent_history = request.serviceHistory[-5:] if request.serviceHistory else []
    service_history_text = "\n".join([
        f"- {h.serviceDate}: {h.actionTaken or h.serviceType} — {h.resolution or h.description}"
        for h in recent_history
    ])

    if request.complaintId:
        if service_history_text:
            try:
                redis_client.setex(f"service_history:{request.complaintId}", 86400, service_history_text)
            except:
                pass
        else:
            try:
                cached_history = redis_client.get(f"service_history:{request.complaintId}")
                if cached_history:
                    service_history_text = cached_history
            except:
                pass

        if request.issueType:
            try:
                redis_client.setex(f"issue_type:{request.complaintId}", 86400, request.issueType)
            except:
                pass
        else:
            try:
                cached_type = redis_client.get(f"issue_type:{request.complaintId}")
                if cached_type:
                    request.issueType = cached_type
            except:
                pass

    # Retrieve previous suggestion from Redis if follow-up mode
    prev_suggestion = None
    if request.aiAttemptCount > 1 and request.complaintId:
        try:
            stored = redis_client.get(f"prev_suggestion:{request.complaintId}")
            if stored:
                try:
                    prev_list = json.loads(stored)
                    prev_suggestion = "\n---\n".join(prev_list)
                except:
                    prev_suggestion = stored
        except:
            pass
    if not prev_suggestion:
        prev_suggestion = request.previousSuggestion

    # Execute modular RAG Flow
    suggestion, final_confidence, predicted_category, top_issue_id = await rag_pipeline.execute(
        query=search_query,
        vehicle_model=request.vehicleModel,
        service_history_text=service_history_text,
        attempt_count=request.aiAttemptCount,
        prev_suggestion=prev_suggestion,
        user_followup=request.userFollowUp,
        issue_type=request.issueType
    )

    # Persistence to local MongoDB and outer Gateway async-ly
    async def persist():
        try:
            # 1. MongoDB Local logging
            mongo_db.queries.insert_one({
                "userId": request.userId,
                "vehicleId": request.vehicleId,
                "vehicleModel": request.vehicleModel or "Unknown",
                "question": search_query,
                "timestamp": datetime.utcnow()
            })
            mongo_db.responses.insert_one({
                "userId": request.userId,
                "vehicleId": request.vehicleId,
                "issueId": f"COMP-{request.complaintId}" if request.complaintId else top_issue_id,
                "answer": suggestion,
                "confidence": final_confidence,
                "status": "ANALYZED",
                "title": request.title or "General",
                "description": search_query,
                "timestamp": datetime.utcnow()
            })
            print("[MongoDB] Successfully logged query and response", flush=True)
        except Exception as e:
            print(f"[MongoDB] Persistence error: {e}", flush=True)

        # 2. External Spring Boot Gateway
        query = await save_query_to_gateway(
            GatewayQueryCreate(
                userId=request.userId,
                vehicleId=request.vehicleId or "Unknown",
                vehicleModel=request.vehicleModel or "Unknown",
                question=search_query
            )
        )
        if query and "id" in query:
            await save_response_to_gateway(
                GatewayResponseCreate(
                    queryId=query["id"],
                    userId=request.userId,
                    vehicleId=request.vehicleId or "Unknown",
                    issueId=f"COMP-{request.complaintId}" if request.complaintId else top_issue_id,
                    answer=suggestion,
                    confidence=final_confidence,
                    status="ANALYZED",
                    title=request.title or "General",
                    description=search_query
                )
            )

    asyncio.create_task(persist())

    res_obj = AIResponse(
        suggestion=suggestion,
        confidence=final_confidence,
        predictedCategory=predicted_category,
        status="SUCCESS"
    )

    # Store interaction in Redis for future follow-ups
    if request.complaintId:
        try:
            stored = redis_client.get(f"prev_suggestion:{request.complaintId}")
            prev_list = []
            if stored:
                try:
                    prev_list = json.loads(stored)
                except:
                    prev_list = [stored]
            
            entry = f"AI: {suggestion}" if request.aiAttemptCount <= 1 else f"User: {request.userFollowUp}\nAI: {suggestion}"
            prev_list.append(entry)
            
            # Keep only the last 2 interactions
            redis_client.setex(f"prev_suggestion:{request.complaintId}", 86400, json.dumps(prev_list[-2:]))
            print(f"[Redis] Stored prev_suggestion key: prev_suggestion:{request.complaintId}", flush=True)
        except Exception as e:
            print(f"[Redis] prev_suggestion store error: {e}", flush=True)

    # Cache response only if it's a valid complete answer (not insufficient/error)
    if "INSUFFICIENT" not in suggestion and "unavailable" not in suggestion:
        try:
            redis_client.setex(cache_key, 3600, res_obj.json())
            print(f"[Redis] Cached response key: {cache_key}", flush=True)
        except Exception as e:
            print(f"[Redis] Cache store error: {e}", flush=True)

    return res_obj

@app.post("/ingest")
async def trigger_ingestion(request: dict, background_tasks: BackgroundTasks):
    from scripts.ingest_documents import ingest_documents
    s3_key = request.get('s3_key')
    file_path = request.get('file_path')
    vehicle_model = request.get('vehicle_model', 'Unknown')
    clear_existing = request.get('clear_existing', False)

    background_tasks.add_task(
        ingest_documents,
        s3_key=s3_key,
        file_path=file_path,
        vehicle_model=vehicle_model,
        clear_existing=clear_existing
    )
    return {"message": "Ingestion started"}

@app.get("/status")
async def get_status():
    return {"status": "active", "integration": "Spring Boot Gateway + MongoDB"}
