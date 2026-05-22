import json
import re
import tiktoken
from typing import Optional, Tuple
from .retriever import QdrantRetriever
from .reranker import Reranker
from .llm_client import GeminiClient

class RAGPipeline:
    def __init__(self, collection_name: str = "ev-manual-index"):
        self.retriever = QdrantRetriever(collection_name)
        self.reranker = Reranker()
        self.llm = GeminiClient()

    def get_token_count(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        try:
            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except:
            return len(text) // 4

    def truncate_by_tokens(self, text: str, max_tokens: int = 1500, model: str = "gpt-3.5-turbo") -> str:
        try:
            enc = tiktoken.encoding_for_model(model)
            tokens = enc.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return enc.decode(tokens[:max_tokens]) + "...[truncated]"
        except:
            return text[:max_tokens * 4]

    def clean_category_name(self, category: str) -> str:
        if not category or category == "UNKNOWN":
            return "General Diagnostic"
        # Remove leading numbers/bullets like "9. ", "Section 2: ", etc.
        cleaned = re.sub(r'^(Section\s*\d+[:\-]?)?(\d+[\.\-\)]\s*)?', '', category, flags=re.IGNORECASE).strip()
        # Truncate if too long
        words = cleaned.split()
        if len(words) > 6:
            return " ".join(words[:6]) + "..."
        return cleaned if cleaned else "General Diagnostic"

    async def execute(
        self,
        query: str,
        vehicle_model: Optional[str] = None,
        service_history_text: str = "",
        attempt_count: int = 1,
        prev_suggestion: Optional[str] = None,
        user_followup: Optional[str] = None,
        issue_type: Optional[str] = None
    ) -> Tuple[str, float, str, str]:
        """
        Coordinates full RAG flow: Retrieval -> Reranking -> LLM Query Generation
        Returns a tuple: (suggestion, confidence, predicted_category, issue_id)
        """
        # 1. Retrieve
        raw_matches = self.retriever.retrieve(query, top_k=8, vehicle_model=vehicle_model)

        # 2. Rerank
        top_matches = self.reranker.rerank(query, raw_matches, top_n=4)

        # Prepare context and identify top attributes
        manual_context = ""
        top_confidence = 0.0
        top_issue_id = "UNKNOWN"
        top_section = "UNKNOWN"

        if top_matches:
            top_match = top_matches[0]
            # Since top_match score is updated during rerank, it serves as confidence reference
            top_confidence = float(top_match.get("score", 0.0))
            metadata = top_match.get("metadata", {})
            top_issue_id = metadata.get("issue_id", "UNKNOWN")
            top_section = metadata.get("section", "UNKNOWN")

            manual_chunks = []
            current_tokens = 0
            for m in top_matches:
                meta = m.get("metadata", {})
                chunk_text = f"SECTION: {meta.get('section', 'General')}\nCONTENT: {meta.get('text', 'No content')}\n"
                chunk_tokens = self.get_token_count(chunk_text)
                if current_tokens + chunk_tokens > 1000:
                    if not manual_chunks:
                        manual_chunks.append(self.truncate_by_tokens(chunk_text, 1000))
                    break
                manual_chunks.append(chunk_text)
                current_tokens += chunk_tokens
            
            manual_context = "---\n".join(manual_chunks)

        # 3. Construct prompt depending on attempt count
        if attempt_count <= 1:
            prompt = f"""You are an EV Diagnostic Assistant helping a driver who is currently experiencing an issue. Your response must be extremely clear, practical, and easy to understand for someone who is not a mechanic.

RULES:
- Base your advice ONLY on the provided MANUAL CONTEXT. Do not invent troubleshooting steps.
- Translate technical manual jargon into simple, plain English that an everyday driver can understand.
- BREVITY: Keep your identified_issue and steps extremely short and crisp (max 2 sentences per field).
- Provide clear, step-by-step instructions on what the driver should do RIGHT NOW.
- BRAND SAFETY: Only mention service centers or technical terms relevant to {vehicle_model or "the vehicle"}.
- If the manual context does not contain a solution or relevant information → return "INSUFFICIENT INFORMATION IN MANUAL CONTEXT".
- Keep it practical: focus on safety and immediate actions.

VEHICLE: {vehicle_model}
COMPLAINT: {query}
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
            prompt = f"""You are an EV Diagnostic Assistant in FOLLOW-UP MODE, chatting with a driver. Make your answer extremely clear, direct, and easy to understand.

RULES:
- Directly answer the driver's FOLLOW-UP QUESTION immediately. Do not repeat the original diagnosis unless it changes.
- BREVITY: Keep your refined_explanation and steps extremely short and crisp (max 2 sentences per field).
- CONSISTENCY: Stay consistent with your previous diagnosis and identified issue unless the driver provides new information that changes the situation.
- BRAND SAFETY: Only mention service centers or technical terms relevant to {vehicle_model or "the vehicle"}.
- Translate any technical jargon into plain English.
- Focus on practical, immediate actions the driver can safely take.
- If the manual doesn't have the answer → return "INSUFFICIENT INFORMATION IN MANUAL CONTEXT".

VEHICLE: {vehicle_model}
COMPLAINT: {query}
SERVICE HISTORY: {service_history_text if service_history_text else "None"}
FOLLOW-UP QUESTION: {user_followup or 'Clarify previous steps'}
PREVIOUS RESPONSE: {self.truncate_by_tokens(str(prev_suggestion), 400) if prev_suggestion else "None"}
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

        # 4. Generate LLM response using Gemini client
        try:
            suggestion = await self.llm.generate_response(prompt)
        except Exception as e:
            print(f"RAG LLM Error: {e}", flush=True)
            suggestion = json.dumps({
                "identified_issue": "AI service unavailable",
                "possible_causes": [],
                "recommended_steps": ["Retry later"]
            })

        # Strip markdown code blocks if the LLM output wrapped the JSON
        if "```" in suggestion:
            suggestion = re.sub(r'```[a-z]*\n?', '', suggestion).strip()
            suggestion = suggestion.replace('```', '').strip()

        # Parse confidence from suggestion if provided
        ai_confidence = 0.0
        try:
            ai_json = json.loads(suggestion)
            ai_confidence = float(ai_json.get("confidence", 0.0))
        except:
            pass

        final_confidence = max(top_confidence, ai_confidence)
        
        # Calculate predicted category
        raw_category = issue_type or (top_section if top_issue_id != "UNKNOWN" else "UNKNOWN")
        predicted_category = self.clean_category_name(raw_category)

        return suggestion, final_confidence, predicted_category, top_issue_id
