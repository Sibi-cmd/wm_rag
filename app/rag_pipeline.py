import json
import logging
import re
import tiktoken
from typing import Optional, Tuple
from rag_core import RAGConfig, RAGPipeline as CoreRAGPipeline

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, collection_name: str = "warehouse-index"):
        config_dict = {
            "embedder": {
                "provider": "sentence-transformers",
                "model": "sentence-transformers/all-mpnet-base-v2"
            },
            "vector_store": {
                "provider": "qdrant",
                "index_name": collection_name,
            },
            "generator": {
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "max_retries": 3,
            },
            "chunker": {
                "provider": "warehouse",
                "chunk_size": 500,
                "chunk_overlap": 50,
            },
            "retrieval": {
                "top_k": 8,
                "rerank": True,
                "rerank_top_k": 4,
                "extra": {
                    "reranker_provider": "warehouse"
                }
            }
        }
        config = RAGConfig.from_dict(config_dict)
        self.core_pipeline = CoreRAGPipeline(config)

    def get_token_count(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        try:
            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except Exception:
            return len(text) // 4

    def truncate_by_tokens(self, text: str, max_tokens: int = 1500, model: str = "gpt-3.5-turbo") -> str:
        try:
            enc = tiktoken.encoding_for_model(model)
            tokens = enc.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return enc.decode(tokens[:max_tokens]) + "...[truncated]"
        except Exception:
            return text[:max_tokens * 4]

    def clean_category_name(self, category: str) -> str:
        if not category or category == "UNKNOWN":
            return "General Logistics"
        # Remove leading numbers/bullets like "9. ", "Section 2: ", etc.
        cleaned = re.sub(r'^(Section\s*\d+[:\-]?)?(\d+[\.\-\)]\s*)?', '', category, flags=re.IGNORECASE).strip()
        # Truncate if too long
        words = cleaned.split()
        if len(words) > 6:
            return " ".join(words[:6]) + "..."
        return cleaned if cleaned else "General Logistics"

    async def execute(
        self,
        query: str,
        document_type: Optional[str] = None,
        sku: Optional[str] = None,
        product_id: Optional[str] = None,
        category: Optional[str] = None,
        warehouse_id: Optional[str] = None,
        zone: Optional[str] = None,
        rack: Optional[str] = None,
        shelf: Optional[str] = None,
        bin: Optional[str] = None,
        audit_trail_text: str = "",
        attempt_count: int = 1,
        prev_response: Optional[str] = None,
        follow_up_question: Optional[str] = None,
        priority: Optional[str] = None
    ) -> Tuple[str, float, str, str]:
        """
        Coordinates full RAG flow: Retrieval -> Reranking -> LLM Query Generation
        Returns a tuple: (suggestion, confidence, predicted_category, ocr_document_id)
        """
        filters = {}
        if document_type:
            filters["document_type"] = document_type
        if sku:
            filters["sku"] = sku
        if product_id:
            filters["product_id"] = product_id
        if category:
            filters["category"] = category
        if warehouse_id:
            filters["warehouse_id"] = warehouse_id
        if zone:
            filters["zone"] = zone
        if rack:
            filters["rack"] = rack
        if shelf:
            filters["shelf"] = shelf
        if bin:
            filters["bin"] = bin

        # 1. Retrieve initial candidates using embedder + vector store components
        query_vector = self.core_pipeline.embedder.embed(query)
        raw_results = self.core_pipeline.vector_store.search(
            vector=query_vector,
            top_k=8,
            filters=filters
        )

        # 2. Rerank using the core pipeline's custom warehouse reranker
        top_results = self.core_pipeline.reranker.rerank(query, raw_results, top_k=4)

        # Convert back to dicts to maintain exact output structure for debugging/compatibility
        raw_matches = [
            {
                "id": r.chunk_id,
                "score": r.score,
                "metadata": r.metadata
            }
            for r in raw_results
        ]
        top_matches = [
            {
                "id": r.chunk_id,
                "score": r.score,
                "metadata": r.metadata
            }
            for r in top_results
        ]

        logger.debug(f"raw_matches count: {len(raw_matches)}")
        logger.debug(f"top_matches count: {len(top_matches)}")

        # Prepare context and identify top attributes
        manual_context = ""
        top_confidence = 0.0
        top_ocr_doc_id = "UNKNOWN"
        top_section = "UNKNOWN"

        if top_matches:
            top_match = top_matches[0]
            top_confidence = float(top_match.get("score", 0.0))
            metadata = top_match.get("metadata", {})
            top_ocr_doc_id = metadata.get("ocr_document_id", "UNKNOWN")
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
            prompt = f"""You are a Warehouse Knowledge Assistant helping a warehouse operator locate inventory, trace shipments, inspect RFQs, verify purchase orders, or troubleshoot logistics processes. Your response must be extremely clear, practical, and easy to understand for warehouse staff.

RULES:
- Base your advice ONLY on the provided DOCUMENT CONTEXT. Do not invent details or facts.
- Translate complex procedural terminology into simple, direct operational instructions.
- BREVITY: Keep your summary and steps extremely short and crisp (max 2 sentences per field).
- Provide clear, step-by-step instructions on what the operator should do next.
- SAFETY & STANDARDS: Ensure actions comply with standard warehouse safety guidelines.
- If the document context does not contain a solution or relevant information → return "INSUFFICIENT INFORMATION IN DOCUMENT CONTEXT".

DOCUMENT TYPE: {document_type or "General"}
QUERY: {query}
AUDIT TRAIL: {audit_trail_text if audit_trail_text else "None"}
DOCUMENT CONTEXT: {manual_context if manual_context else "No technical entries found."}

OUTPUT FORMAT: STRICT JSON — keep all values CLEAR and ACTIONABLE.
{{
  "analysis_summary": "A simple explanation of the situation, query details, or status.",
  "audit_context": "How the audit trail relates to this query (if applicable).",
  "possible_issues": ["Issue 1 described simply", "Issue 2 described simply"],
  "recommended_actions": ["Action 1: Immediate practical step for the operator.", "Action 2: Next step.", "Action 3: Escalation path if unresolved."],
  "document_reference": ["Section X.X or Page Y"],
  "confidence": 0.0
}}
"""
        else:
            prompt = f"""You are a Warehouse Knowledge Assistant in FOLLOW-UP MODE, chatting with a warehouse operator. Make your answer extremely clear, direct, and easy to understand.

RULES:
- Directly answer the operator's FOLLOW-UP QUESTION immediately. Do not repeat the original diagnosis unless it has changed.
- BREVITY: Keep your explanation and steps extremely short and crisp (max 2 sentences per field).
- CONSISTENCY: Stay consistent with your previous response and analysis unless new details change the situation.
- Translate any technical manual or logistics jargon into plain English.
- Focus on practical, immediate actions the operator can safely take.
- If the document context doesn't have the answer → return "INSUFFICIENT INFORMATION IN DOCUMENT CONTEXT".

DOCUMENT TYPE: {document_type or "General"}
QUERY: {query}
AUDIT TRAIL: {audit_trail_text if audit_trail_text else "None"}
FOLLOW-UP QUESTION: {follow_up_question or 'Clarify previous steps'}
PREVIOUS RESPONSE: {self.truncate_by_tokens(str(prev_response), 400) if prev_response else "None"}
DOCUMENT CONTEXT: {manual_context if manual_context else "No entries found."}

OUTPUT FORMAT: STRICT JSON — keep all values CLEAR and ACTIONABLE.
{{
  "refined_explanation": "A simple, easy-to-understand clarification.",
  "audit_trail_update": "Any relevant update based on audit history.",
  "possible_issues": ["Issue 1 described simply", "Issue 2 described simply"],
  "recommended_actions": ["Action 1: Immediate step.", "Action 2: Follow-up action.", "Action 3: Escalation/Safety step."],
  "document_reference": ["Section X.X or Page Y"],
  "confidence": 0.0
}}
"""

        # 4. Generate LLM response using core pipeline's Gemini generator
        try:
            suggestion = await self.core_pipeline.agenerate(prompt)
        except Exception as e:
            logger.error(f"RAG LLM Error: {e}", exc_info=True)
            if "API key not found" in str(e) or "not set" in str(e):
                logger.warning("Running in MOCK LOCAL MODE (no GEMINI_API_KEY). Returning retrieved context mock JSON.")
                suggestion = json.dumps({
                    "analysis_summary": f"[MOCK LOCAL MODE] Found matching document {top_ocr_doc_id} under section '{top_section}'. Set GEMINI_API_KEY in .env to get a real LLM synthesized answer.",
                    "possible_issues": [f"Issues related to: {top_section}"],
                    "recommended_actions": [
                        "Verify terminal connection.",
                        "Add GEMINI_API_KEY to your .env file to enable live Gemini synthesis."
                    ],
                    "document_reference": [f"Section: {top_section}"],
                    "confidence": top_confidence
                })
            else:
                suggestion = json.dumps({
                    "analysis_summary": "AI service temporarily unavailable. Please retry shortly.",
                    "possible_issues": [str(e)],
                    "recommended_actions": ["Retry the request after a short wait."]
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
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.debug(f"Could not parse AI confidence from response: {e}")

        final_confidence = max(top_confidence, ai_confidence)
        
        # Calculate predicted category
        raw_category = top_section if top_ocr_doc_id != "UNKNOWN" else "UNKNOWN"
        predicted_category = self.clean_category_name(raw_category)

        return suggestion, final_confidence, predicted_category, top_ocr_doc_id
