"""
rag_core.api
~~~~~~~~~~~~
standalone FastAPI-based RAG Middleman service.
Provides clean REST API endpoints for external applications (Django / FastAPI).
Also exposes an ultra-premium visual administration dashboard at root.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .router import RAGRouter

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_core.api")

# Create FastAPI app
app = FastAPI(
    title="RAG Middleman Service",
    description="Secure, isolated dynamic routing for multi-tenant RAG systems.",
    version="1.0.0",
)

# Initialize RAG Router
router = RAGRouter()


# ------------------------------------------------------------------
# Request & Response Schemas
# ------------------------------------------------------------------

class SchemaMapping(BaseModel):
    table_name: str = Field(default="documents", description="The table name in the relational DB to query.")
    text_columns: list[str] = Field(default_factory=lambda: ["text"], description="List of columns containing text to chunk & embed.")
    metadata_columns: list[str] = Field(default_factory=lambda: ["source", "section"], description="List of columns to preserve as metadata.")


class AskRequest(BaseModel):
    db_url: Optional[str] = Field(None, description="Connection URL (e.g., sqlite:///db.sqlite). Can be omitted if db_name is a registered alias.")
    db_name: str = Field(..., description="Unique database name or registered alias key.")
    query: str = Field(..., description="The question or search query.")
    top_k: Optional[int] = Field(None, description="Number of context chunks to retrieve.")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters to apply (e.g. {'inspector_id': '123'}).")
    schema_mapping: Optional[SchemaMapping] = Field(None, description="Custom mapping for database schema.")
    force_sync: bool = Field(False, description="Set to true to force re-indexing the database.")


class AskResponse(BaseModel):
    answer: str = Field(..., description="The synthesized LLM response.")
    sources: list[str] = Field(default_factory=list, description="Unique sources used in the context.")
    confidence: float = Field(0.0, description="Highest similarity score from the top matching chunk.")
    chunks_used: int = Field(0, description="Number of context chunks sent to the LLM.")
    db_name: str = Field(..., description="Database name queried.")


class SyncRequest(BaseModel):
    db_url: Optional[str] = Field(None, description="Connection URL. Can be omitted if db_name is a registered alias.")
    db_name: str = Field(..., description="Unique database name or registered alias key.")
    schema_mapping: Optional[SchemaMapping] = Field(None, description="Custom mapping for database schema.")


# ------------------------------------------------------------------
# REST API Endpoints
# ------------------------------------------------------------------

@app.post("/ask", response_model=AskResponse, tags=["RAG Services"])
def ask(payload: AskRequest) -> AskResponse:
    """Submit a query to be routed to a specific database for isolated RAG generation."""
    logger.info(f"API: Received ask request for tenant '{payload.db_name}'")
    
    mapping_dict = payload.schema_mapping.model_dump() if payload.schema_mapping else None
    
    try:
        response = router.ask_tenant(
            db_url=payload.db_url,
            db_name=payload.db_name,
            query=payload.query,
            top_k=payload.top_k,
            filters=payload.filters,
            schema_mapping=mapping_dict,
            force_sync=payload.force_sync,
        )
        return AskResponse(
            answer=response.text,
            sources=response.sources,
            confidence=response.confidence,
            chunks_used=response.chunks_used,
            db_name=payload.db_name,
        )
    except Exception as e:
        logger.error(f"API Error processing /ask for tenant '{payload.db_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sync", tags=["Maintenance"])
def sync(payload: SyncRequest):
    """Trigger manual re-indexing for a specific database to refresh the RAG cache."""
    logger.info(f"API: Triggering manual database sync for tenant '{payload.db_name}'")
    
    mapping_dict = payload.schema_mapping.model_dump() if payload.schema_mapping else None
    
    try:
        # Fetching or creating pipeline will trigger synchronization
        router.get_pipeline(
            db_url=payload.db_url,
            db_name=payload.db_name,
            schema_mapping=mapping_dict,
            force_sync=True,
        )
        return {
            "status": "success",
            "message": f"Database index for '{payload.db_name}' successfully rebuilt and cached.",
        }
    except Exception as e:
        logger.error(f"API Error processing /sync for tenant '{payload.db_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", tags=["Monitoring"])
def health():
    """Service health status check."""
    return {
        "status": "healthy",
        "service": "RAG Middleman",
        "version": "1.0.0",
        "base_model": router.config.get("base_rag_config", "configs/rag_config.yaml"),
    }


@app.get("/stats", tags=["Monitoring"])
def stats():
    """Retrieve connection pool and cache utilization metrics."""
    return router.get_stats()


# ------------------------------------------------------------------
# Interactive Web Dashboard
# ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
def get_dashboard(request: Request):
    """Serve an ultra-premium visual administration dashboard for RAG management."""
    stats_data = router.get_stats()
    active_tenants = stats_data.get("active_tenants", [])
    active_count = len(active_tenants)
    max_size = stats_data.get("max_cache_size", 50)
    
    tenants_html = ""
    if not active_tenants:
        tenants_html = """
        <div class="empty-state">
            <div class="empty-icon">📭</div>
            <p>No active database connections in memory cache.</p>
            <span class="sub">Active databases will appear here automatically when queried.</span>
        </div>
        """
    else:
        for tenant in active_tenants:
            name = tenant["db_name"]
            ago = tenant["last_active_seconds_ago"]
            time_str = "Just now" if ago < 5 else f"{ago}s ago"
            tenants_html += f"""
            <div class="tenant-card">
                <div class="tenant-meta">
                    <span class="tenant-badge">ACTIVE</span>
                    <span class="tenant-title">{name}</span>
                </div>
                <div class="tenant-time">Last active: <strong>{time_str}</strong></div>
            </div>
            """

    # Premium dark glassmorphism dashboard in single block
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG Middleman Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            :root {{
                --bg: #0b0f19;
                --primary: #4f46e5;
                --primary-glow: rgba(79, 70, 229, 0.4);
                --success: #10b981;
                --text-main: #f3f4f6;
                --text-muted: #9ca3af;
                --card-bg: rgba(17, 24, 39, 0.7);
                --border: rgba(255, 255, 255, 0.08);
            }}
            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }}
            body {{
                font-family: 'Plus Jakarta Sans', sans-serif;
                background-color: var(--bg);
                color: var(--text-main);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                overflow-x: hidden;
            }}
            /* Sleek Animated Gradients in Background */
            body::before {{
                content: '';
                position: absolute;
                top: -20%;
                left: -10%;
                width: 600px;
                height: 600px;
                background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, rgba(0,0,0,0) 70%);
                z-index: -1;
                filter: blur(80px);
            }}
            body::after {{
                content: '';
                position: absolute;
                bottom: -10%;
                right: -10%;
                width: 600px;
                height: 600px;
                background: radial-gradient(circle, rgba(16, 185, 129, 0.08) 0%, rgba(0,0,0,0) 70%);
                z-index: -1;
                filter: blur(80px);
            }}
            header {{
                padding: 2rem 4rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid var(--border);
                backdrop-filter: blur(10px);
                background: rgba(11, 15, 25, 0.6);
            }}
            .logo-section {{
                display: flex;
                align-items: center;
                gap: 1rem;
            }}
            .logo-icon {{
                background: linear-gradient(135deg, var(--primary) 0%, #818cf8 100%);
                width: 45px;
                height: 45px;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.5rem;
                box-shadow: 0 4px 20px var(--primary-glow);
                animation: float 4s ease-in-out infinite;
            }}
            .logo-text h1 {{
                font-family: 'Outfit', sans-serif;
                font-size: 1.5rem;
                font-weight: 700;
                background: linear-gradient(135deg, #ffffff 0%, #d1d5db 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            .logo-text p {{
                font-size: 0.8rem;
                color: var(--text-muted);
            }}
            .health-badge {{
                display: flex;
                align-items: center;
                gap: 0.5rem;
                background: rgba(16, 185, 129, 0.1);
                border: 1px solid rgba(16, 185, 129, 0.2);
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-size: 0.85rem;
                font-weight: 600;
                color: var(--success);
            }}
            .pulse {{
                width: 8px;
                height: 8px;
                background-color: var(--success);
                border-radius: 50%;
                box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
                animation: pulse 1.5s infinite;
            }}
            main {{
                flex: 1;
                padding: 3rem 4rem;
                max-width: 1400px;
                margin: 0 auto;
                width: 100%;
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: 3rem;
            }}
            .panel {{
                background: var(--card-bg);
                border: 1px solid var(--border);
                border-radius: 24px;
                padding: 2.5rem;
                backdrop-filter: blur(20px);
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
            }}
            .panel-header {{
                margin-bottom: 2rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .panel-header h2 {{
                font-family: 'Outfit', sans-serif;
                font-size: 1.6rem;
                font-weight: 600;
                letter-spacing: -0.5px;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 1.5rem;
                margin-bottom: 2.5rem;
            }}
            .stat-card {{
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 1.5rem;
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
                transition: transform 0.3s ease, border-color 0.3s ease;
            }}
            .stat-card:hover {{
                transform: translateY(-4px);
                border-color: rgba(99, 102, 241, 0.3);
            }}
            .stat-label {{
                font-size: 0.85rem;
                color: var(--text-muted);
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .stat-val {{
                font-size: 2rem;
                font-weight: 700;
                color: #ffffff;
                font-family: 'Outfit', sans-serif;
            }}
            .endpoint-section h3 {{
                font-family: 'Outfit', sans-serif;
                font-size: 1.2rem;
                margin-bottom: 1rem;
                color: #e5e7eb;
            }}
            .endpoint-row {{
                background: rgba(255, 255, 255, 0.02);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 1rem 1.5rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 0.8rem;
                transition: all 0.2s ease;
            }}
            .endpoint-row:hover {{
                background: rgba(255, 255, 255, 0.04);
            }}
            .method-badge {{
                padding: 0.3rem 0.6rem;
                border-radius: 6px;
                font-size: 0.75rem;
                font-weight: 700;
                letter-spacing: 0.5px;
            }}
            .post-badge {{
                background: rgba(99, 102, 241, 0.15);
                color: #818cf8;
                border: 1px solid rgba(99, 102, 241, 0.3);
            }}
            .get-badge {{
                background: rgba(16, 185, 129, 0.15);
                color: #34d399;
                border: 1px solid rgba(16, 185, 129, 0.3);
            }}
            .endpoint-path {{
                font-family: monospace;
                font-size: 0.95rem;
                color: #f3f4f6;
                font-weight: 600;
                margin-left: 1rem;
            }}
            .endpoint-desc {{
                font-size: 0.85rem;
                color: var(--text-muted);
            }}
            .tenant-card {{
                background: rgba(255, 255, 255, 0.02);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 1.25rem 1.5rem;
                margin-bottom: 1rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
                transition: all 0.3s ease;
            }}
            .tenant-card:hover {{
                border-color: rgba(99, 102, 241, 0.3);
                background: rgba(255, 255, 255, 0.04);
            }}
            .tenant-badge {{
                background: rgba(16, 185, 129, 0.1);
                color: var(--success);
                font-size: 0.7rem;
                font-weight: 700;
                padding: 0.2rem 0.5rem;
                border-radius: 6px;
                margin-right: 0.8rem;
            }}
            .tenant-title {{
                font-weight: 600;
                font-size: 0.95rem;
            }}
            .tenant-time {{
                font-size: 0.85rem;
                color: var(--text-muted);
            }}
            .empty-state {{
                text-align: center;
                padding: 4rem 2rem;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 1rem;
            }}
            .empty-icon {{
                font-size: 3rem;
                animation: pulse 2s infinite;
            }}
            .empty-state p {{
                font-weight: 600;
                color: #e5e7eb;
            }}
            .empty-state .sub {{
                font-size: 0.8rem;
                color: var(--text-muted);
            }}
            .btn-docs {{
                background: linear-gradient(135deg, var(--primary) 0%, #6366f1 100%);
                color: #ffffff;
                text-decoration: none;
                padding: 0.6rem 1.2rem;
                border-radius: 10px;
                font-size: 0.85rem;
                font-weight: 600;
                box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }}
            .btn-docs:hover {{
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(79, 70, 229, 0.5);
            }}
            footer {{
                text-align: center;
                padding: 2rem;
                font-size: 0.8rem;
                color: var(--text-muted);
                border-top: 1px solid var(--border);
                margin-top: auto;
            }}
            @keyframes pulse {{
                0% {{
                    box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4);
                }}
                70% {{
                    box-shadow: 0 0 0 10px rgba(16, 185, 129, 0);
                }}
                100% {{
                    box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
                }}
            }}
            @keyframes float {{
                0%, 100% {{ transform: translateY(0); }}
                50% {{ transform: translateY(-6px); }}
            }}
            @media (max-width: 1024px) {{
                main {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body>
        <header>
            <div class="logo-section">
                <div class="logo-icon">🤖</div>
                <div class="logo-text">
                    <h1>RAG Middleman Service</h1>
                    <p>Secure Enterprise Routing Engine</p>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 1.5rem;">
                <a href="/docs" class="btn-docs" target="_blank">Interactive Swagger Docs</a>
                <div class="health-badge">
                    <div class="pulse"></div>
                    <span>SYSTEM ONLINE</span>
                </div>
            </div>
        </header>

        <main>
            <!-- Left Side: System Details and Endpoints -->
            <div class="panel">
                <div class="panel-header">
                    <h2>RAG Core Overview</h2>
                </div>

                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Cached Pipelines</div>
                        <div class="stat-val">{active_count} <span style="font-size: 1rem; font-weight:400; color:var(--text-muted);">/ {max_size}</span></div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">LLM Provider</div>
                        <div class="stat-val" style="font-size: 1.4rem; padding-top:0.4rem;">OpenAI (gpt-4o-mini)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Embedder Dimensions</div>
                        <div class="stat-val">768</div>
                    </div>
                </div>

                <div class="endpoint-section">
                    <h3>Service API Endpoints</h3>
                    
                    <div class="endpoint-row">
                        <div>
                            <span class="method-badge post-badge">POST</span>
                            <span class="endpoint-path">/ask</span>
                        </div>
                        <div class="endpoint-desc">Dynamically queries a specific database and returns RAG response</div>
                    </div>

                    <div class="endpoint-row">
                        <div>
                            <span class="method-badge post-badge">POST</span>
                            <span class="endpoint-path">/sync</span>
                        </div>
                        <div class="endpoint-desc">Triggers manual re-indexing to refresh the cache for a tenant</div>
                    </div>

                    <div class="endpoint-row">
                        <div>
                            <span class="method-badge get-badge">GET</span>
                            <span class="endpoint-path">/health</span>
                        </div>
                        <div class="endpoint-desc">Returns health status and loaded model specifications</div>
                    </div>

                    <div class="endpoint-row">
                        <div>
                            <span class="method-badge get-badge">GET</span>
                            <span class="endpoint-path">/stats</span>
                        </div>
                        <div class="endpoint-desc">Fetches connection pool statistics and LRU utilization metrics</div>
                    </div>
                </div>
            </div>

            <!-- Right Side: Active Tenants Memory Cache -->
            <div class="panel">
                <div class="panel-header">
                    <h2>Active Pools ({active_count})</h2>
                </div>
                <div class="tenants-list">
                    {tenants_html}
                </div>
            </div>
        </main>

        <footer>
            Built with Antigravity AI Engine &copy; 2026. Standalone Plug-and-Play RAG Architecture.
        </footer>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)
