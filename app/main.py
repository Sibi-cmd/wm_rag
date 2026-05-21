from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from typing import List, Optional
import httpx
import json
import hashlib
import asyncio
import tiktoken
import re

from scripts.ingest_from_s3 import ingest_from_s3
from .models import (
    AIRequest, AIResponse, GatewayQueryCreate, GatewayResponseCreate,
    QueryCreate, QueryResponse
)
from .database import redis_client

load_dotenv()

# Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
SPRING_BOOT_URL = os.getenv("SPRING_BOOT_URL", "http://localhost:8081") # Gateway URL

mistral_client = None
embeddings_model = None
pinecone_index = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global mistral_client, embeddings_model, pinecone_index
    print("Starting AI service...", flush=True)
    from mistralai.client import Mistral
    mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    from sentence_transformers import SentenceTransformer
    embeddings_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    from pinecone import Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index("ev-manual-index")
    print("Models loaded successfully! AI Service is ready.", flush=True)
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
    url = f"{SPRING_BOOT_URL}/api/ai/queries"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=query_data.dict(), timeout=5.0)
            return response.json() if response.status_code == 200 else None
        except: return None

async def save_response_to_gateway(response_data: GatewayResponseCreate):
    url = f"{SPRING_BOOT_URL}/api/ai/responses"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=response_data.dict(), timeout=5.0)
            return response.json() if response.status_code == 200 else None
        except: return None

def get_token_count(text, model="gpt-3.5-turbo"):
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except: return len(text) // 4

def truncate_by_tokens(text, max_tokens=1500, model="gpt-3.5-turbo"):
    try:
        enc = tiktoken.encoding_for_model(model)
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens: return text
        return enc.decode(tokens[:max_tokens]) + "...[truncated]"
    except: return text[:max_tokens * 4]

def calculate_rerank_score(query, match):
    sim_score = match.score
    metadata = match.metadata
    text = metadata.get('text', '').lower()
    query_lower = (query or "").lower()
    query_words = set(re.findall(r'\w+', query_lower))
    text_words = set(re.findall(r'\w+', text))
    overlap = len(query_words.intersection(text_words)) / max(len(query_words), 1)
    section = metadata.get('section', '').lower()
    section_match = 1.0 if section in query_lower and section else 0.0
    severity = metadata.get('severity', 'low').lower()
    severity_boost = 1.0 if severity == 'high' else (0.5 if severity == 'medium' else 0.0)
    return (0.6 * sim_score) + (0.2 * overlap) + (0.1 * section_match) + (0.1 * severity_boost)

def clean_category_name(category: str) -> str:
    if not category or category == "UNKNOWN": return "General Diagnostic"
    # Remove leading numbers/bullets like "9. ", "Section 2: ", etc.
    cleaned = re.sub(r'^(Section\s*\d+[:\-]?)?(\d+[\.\-\)]\s*)?', '', category, flags=re.IGNORECASE).strip()
    # Truncate if too long
    words = cleaned.split()
    if len(words) > 6: return " ".join(words[:6]) + "..."
    return cleaned if cleaned else "General Diagnostic"

# --- Main API Endpoints ---

@app.post("/api/ai/analyze", response_model=AIResponse)
async def analyze_complaint(request: AIRequest):
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
            print(f"[Redis] Cache hit for complaint {request.complaintId}")
            return AIResponse(**json.loads(cached_data))
    except: pass

    # 1. RAG: Search Pinecone & Rerank
    manual_context = ""
    top_confidence = 0.0
    top_issue_id = "UNKNOWN"
    
    try:
        embedding = embeddings_model.encode(search_query).tolist()
        pc_response = pinecone_index.query(vector=embedding, top_k=8, include_metadata=True, filter={"vehicle_model": request.vehicleModel} if request.vehicleModel else None)
        if (not hasattr(pc_response, 'matches') or not pc_response.matches) and request.vehicleModel:
            pc_response = pinecone_index.query(vector=embedding, top_k=8, include_metadata=True)
        
        if hasattr(pc_response, 'matches') and pc_response.matches:
            # Filter out generic headers like 'CONTENTS', 'INDEX', etc.
            filtered_matches = [m for m in pc_response.matches if m.metadata and m.metadata.get("section", "").upper() not in ["CONTENTS", "INDEX", "PREFACE"]]
            target_matches = filtered_matches if filtered_matches else pc_response.matches
            
            scored_matches = sorted([(calculate_rerank_score(search_query, m), m) for m in target_matches if m.metadata], key=lambda x: x[0], reverse=True)
            top_4_matches = [x[1] for x in scored_matches[:4]]
            if top_4_matches:
                top_match = top_4_matches[0]
                top_confidence, top_issue_id = float(top_match.score), top_match.metadata.get("issue_id", "UNKNOWN")
                manual_chunks = []
                current_tokens = 0
                for m in top_4_matches:
                    chunk_text = f"SECTION: {m.metadata.get('section', 'General')}\nCONTENT: {m.metadata.get('text', 'No content')}\n"
                    chunk_tokens = get_token_count(chunk_text)
                    if current_tokens + chunk_tokens > 1000:
                        if not manual_chunks: manual_chunks.append(truncate_by_tokens(chunk_text, 1000))
                        break
                    manual_chunks.append(chunk_text); current_tokens += chunk_tokens
                manual_context = "---\n".join(manual_chunks)
    except Exception as e: print(f"RAG Error: {e}")

    # Service History: Prioritize Spring Boot, fallback to Redis
    recent_history = request.serviceHistory[-5:] if request.serviceHistory else []
    service_history_text = "\n".join([
        f"- {h.serviceDate}: {h.actionTaken or h.serviceType} — {h.resolution or h.description}"
        for h in recent_history
    ])

    if request.complaintId:
        if service_history_text:
            try: redis_client.setex(f"service_history:{request.complaintId}", 86400, service_history_text)
            except: pass
        else:
            try:
                cached_history = redis_client.get(f"service_history:{request.complaintId}")
                if cached_history: service_history_text = cached_history
            except: pass

        if request.issueType:
            try: redis_client.setex(f"issue_type:{request.complaintId}", 86400, request.issueType)
            except: pass
        else:
            try:
                cached_type = redis_client.get(f"issue_type:{request.complaintId}")
                if cached_type: request.issueType = cached_type
            except: pass

    if request.aiAttemptCount <= 1:
        prompt = f"""You are an EV Diagnostic Assistant helping a driver who is currently experiencing an issue. Your response must be extremely clear, practical, and easy to understand for someone who is not a mechanic.

RULES:
- Base your advice ONLY on the provided MANUAL CONTEXT. Do not invent troubleshooting steps.
- Translate technical manual jargon into simple, plain English that an everyday driver can understand.
- BREVITY: Keep your identified_issue and steps extremely short and crisp (max 2 sentences per field).
- Provide clear, step-by-step instructions on what the driver should do RIGHT NOW.
- BRAND SAFETY: Only mention service centers or technical terms relevant to {request.vehicleModel or "the vehicle"}.
- If the manual context does not contain a solution or relevant information → return "INSUFFICIENT INFORMATION IN MANUAL CONTEXT".
- Keep it practical: focus on safety and immediate actions.

VEHICLE: {request.vehicleModel} | ID: {request.vehicleId}
COMPLAINT: {request.title} - {request.description}
SERVICE HISTORY: {service_history_text if service_history_text else "None"}
MANUAL CONTEXT: {manual_context if manual_context else "No technical entries found."}

OUTPUT FORMAT: STRICT JSON — keep all values CLEAR and ACTIONABLE.
{{
  "identified_issue": "A simple explanation of what is likely happening, in plain English.",
  "service_history_analysis": "How past service relates to this issue (if applicable).",
  "possible_causes": ["Cause 1 described simply", "Cause 2 described simply"],
  "recommended_steps": ["Step 1: Immediate practical action for the driver.", "Step 2: Next practical step.", "Step 3: When to call for service."],
  "manual_reference": ["Section X.X"],
  "confidence": 0.0
}}
"""
    else:
        # Attempt 2+: Retrieve previous suggestion from Redis
        prev_suggestion = None
        if request.complaintId:
            try: 
                stored = redis_client.get(f"prev_suggestion:{request.complaintId}")
                if stored:
                    try:
                        prev_list = json.loads(stored)
                        prev_suggestion = "\n---\n".join(prev_list)
                    except:
                        prev_suggestion = stored
            except: pass
        if not prev_suggestion: prev_suggestion = request.previousSuggestion

        prompt = f"""You are an EV Diagnostic Assistant in FOLLOW-UP MODE, chatting with a driver. Make your answer extremely clear, direct, and easy to understand.

RULES:
- Directly answer the driver's FOLLOW-UP QUESTION immediately. Do not repeat the original diagnosis unless it changes.
- BREVITY: Keep your refined_explanation and steps extremely short and crisp (max 2 sentences per field).
- CONSISTENCY: Stay consistent with your previous diagnosis and identified issue unless the driver provides new information that changes the situation.
- BRAND SAFETY: Only mention service centers or technical terms relevant to {request.vehicleModel or "the vehicle"}.
- Translate any technical jargon into plain English.
- Focus on practical, immediate actions the driver can safely take.
- If the manual doesn't have the answer → return "INSUFFICIENT INFORMATION IN MANUAL CONTEXT".

VEHICLE: {request.vehicleModel} | ID: {request.vehicleId}
COMPLAINT: {request.title} - {request.description}
SERVICE HISTORY: {service_history_text if service_history_text else "None"}
FOLLOW-UP QUESTION: {request.userFollowUp or 'Clarify previous steps'}
PREVIOUS RESPONSE: {truncate_by_tokens(str(prev_suggestion), 400)}
MANUAL CONTEXT: {manual_context if manual_context else "No entries found."}

OUTPUT FORMAT: STRICT JSON — keep all values CLEAR and ACTIONABLE.
{{
  "refined_explanation": "A simple, easy-to-understand clarification.",
  "service_history_update": "Any relevant update based on service history.",
  "possible_causes": ["Cause 1 described simply", "Cause 2 described simply"],
  "recommended_steps": ["Step 1: Practical action.", "Step 2: Next step.", "Step 3: When to seek help."],
  "manual_reference": ["Section X.X"],
  "confidence": 0.0
}}
"""

    # 3. Call Mistral AI
    suggestion = None
    for attempt in range(1, 4):
        try:
            ai_res = await mistral_client.chat.complete_async(model="mistral-small-latest", messages=[{"role": "user", "content": prompt}])
            suggestion = ai_res.choices[0].message.content
            break
        except Exception as e:
            print(f"Mistral API Error (Attempt {attempt}): {e}")
            if "429" in str(e) and attempt < 3: await asyncio.sleep([2, 4][attempt-1])
            else: break

    if suggestion is None:
        suggestion = json.dumps({"identified_issue": "AI service unavailable", "possible_causes": [], "recommended_steps": ["Retry later"]})
    
    # Clean up markdown code blocks if present
    if "```" in suggestion:
        suggestion = re.sub(r'```[a-z]*\n?', '', suggestion).strip()
        suggestion = suggestion.replace('```', '').strip()

    # 4. Persistence & Cache
    # Parse AI's self-reported confidence if available
    ai_confidence = 0.0
    try:
        ai_json = json.loads(suggestion)
        ai_confidence = float(ai_json.get("confidence", 0.0))
    except: pass
    
    final_confidence = max(top_confidence, ai_confidence)
    raw_category = request.issueType or (top_match.metadata.get("section") if top_issue_id != "UNKNOWN" else "UNKNOWN")
    final_category = clean_category_name(raw_category)

    async def persist():
        query = await save_query_to_gateway(GatewayQueryCreate(userId=request.userId, vehicleId=request.vehicleId, vehicleModel=request.vehicleModel or "Unknown", question=search_query))
        if query and "id" in query:
            await save_response_to_gateway(GatewayResponseCreate(queryId=query["id"], userId=request.userId, vehicleId=request.vehicleId, issueId=f"COMP-{request.complaintId}" if request.complaintId else top_issue_id, answer=suggestion, confidence=final_confidence, status="ANALYZED", title=request.title or "General", description=search_query))

    asyncio.create_task(persist())
    res_obj = AIResponse(
        suggestion=suggestion,
        confidence=final_confidence,
        predictedCategory=final_category,
        status="SUCCESS"
    )
    # Always store prev_suggestion so follow-ups can retrieve context
    if request.complaintId:
        try:
            stored = redis_client.get(f"prev_suggestion:{request.complaintId}")
            prev_list = []
            if stored:
                try: prev_list = json.loads(stored)
                except: prev_list = [stored]
            
            entry = f"AI: {suggestion}" if request.aiAttemptCount <= 1 else f"User: {request.userFollowUp}\nAI: {suggestion}"
            prev_list.append(entry)
            
            # Keep only the last 2 interactions
            redis_client.setex(f"prev_suggestion:{request.complaintId}", 86400, json.dumps(prev_list[-2:]))
            print(f"[Redis] Stored prev_suggestion list (len={len(prev_list[-2:])}) → key: prev_suggestion:{request.complaintId}")
        except Exception as e: print(f"[Redis] prev_suggestion store error: {e}")

    # Only cache the full response if it's a valid answer (not insufficient/error)
    if "INSUFFICIENT" not in suggestion and "unavailable" not in suggestion:
        try:
            redis_client.setex(cache_key, 3600, res_obj.json())
            print(f"[Redis] Cached response → key: {cache_key}")
        except Exception as e: print(f"[Redis] Cache store error: {e}")

    return res_obj

@app.post("/ingest")
async def trigger_s3_ingestion(request: dict, background_tasks: BackgroundTasks):
    background_tasks.add_task(ingest_from_s3, request['s3_key'], request['vehicle_model'], request.get('clear_existing', False))
    return {"message": "Ingestion started"}

@app.get("/status")
async def get_status():
    return {"status": "active", "integration": "Spring Boot Gateway"}
