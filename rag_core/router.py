"""
rag_core.router
~~~~~~~~~~~~~~~
Dynamic Multi-Database RAG Router. Routes questions to isolated databases
using a connection pool and index caching.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from sqlalchemy import create_engine, text

from .config import RAGConfig, VectorStoreConfig
from .pipeline import RAGPipeline
from .types import RAGResponse

logger = logging.getLogger("rag_core.router")


class RAGRouter:
    """Manages dynamic connections and isolated RAGPipelines for multiple databases."""

    def __init__(self, config_path: str = "configs/router_config.yaml") -> None:
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load_config()

        # Get defaults
        router_settings = self.config.get("router", {})
        self.max_cached_pipelines = int(router_settings.get("max_cached_pipelines", 50))
        self.ttl_seconds = int(router_settings.get("ttl_seconds", 1800))
        self.local_index_base_dir = Path(router_settings.get("local_index_base_dir", "./vector_dbs"))
        self.local_index_base_dir.mkdir(parents=True, exist_ok=True)

        self._cache: Dict[str, Dict[str, Any]] = {}  # tenant_key -> {pipeline, last_used, db_url, db_name}

    def load_config(self) -> None:
        """Load configuration details and aliases."""
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as fh:
                self.config = yaml.safe_load(fh) or {}
        else:
            logger.warning(f"Router config not found at {self.config_path}. Using default settings.")
            self.config = {}

    def _get_connection_url(self, db_url: Optional[str], db_name: str) -> str:
        """Resolve dynamic database connection URL or registry alias."""
        # 1. Check if db_name is registered in the connection aliases
        aliases = self.config.get("connections", {})
        if db_name in aliases:
            return aliases[db_name]

        # 2. If not an alias and db_url is supplied, use db_url
        if db_url:
            return db_url

        # 3. Fallback
        raise ValueError(
            f"Unable to resolve database connection for db_name='{db_name}'. "
            "Please provide a valid 'db_url' or register the 'db_name' as an alias in router_config.yaml."
        )

    def _get_tenant_key(self, resolved_url: str, db_name: str) -> str:
        """Create a secure unique hash for a database connection."""
        hash_input = f"{resolved_url}||{db_name}"
        return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

    def get_pipeline(
        self,
        db_url: Optional[str],
        db_name: str,
        schema_mapping: Optional[Dict[str, Any]] = None,
        force_sync: bool = False,
    ) -> RAGPipeline:
        """Retrieve or dynamically instantiate an isolated RAGPipeline for the database."""
        resolved_url = self._get_connection_url(db_url, db_name)
        tenant_key = self._get_tenant_key(resolved_url, db_name)

        # LRU Check & update last used timestamp if already cached
        if tenant_key in self._cache:
            entry = self._cache[tenant_key]
            entry["last_used"] = time.time()
            if force_sync:
                logger.info(f"Force-sync requested for tenant '{db_name}'. Rebuilding index.")
                self._sync_database(entry["pipeline"], resolved_url, db_name, schema_mapping)
            return entry["pipeline"]

        # Evict inactive pipelines if cache is full
        self._evict_expired_pipelines()
        self._evict_lru_if_needed()

        # Build pipeline dynamically
        logger.info(f"Instantiating new RAGPipeline for tenant '{db_name}'...")
        pipeline = self._create_pipeline_for_tenant(db_name)

        # Check if vector index files already exist locally on disk
        index_path = self.local_index_base_dir / db_name
        faiss_index_file = index_path / "index.faiss"

        # If index doesn't exist, or if we force sync, populate it from the SQL database
        if not faiss_index_file.exists() or force_sync:
            logger.info(f"Local index files not found or force-sync enabled for '{db_name}'. Indexing source database...")
            self._sync_database(pipeline, resolved_url, db_name, schema_mapping)

        # Add to cache
        self._cache[tenant_key] = {
            "pipeline": pipeline,
            "last_used": time.time(),
            "db_url": resolved_url,
            "db_name": db_name,
        }

        return pipeline

    def _create_pipeline_for_tenant(self, db_name: str) -> RAGPipeline:
        """Build an isolated RAGPipeline instance with custom local FAISS folder."""
        base_config_path = self.config.get("base_rag_config", "configs/rag_config.yaml")
        base_config = RAGConfig.from_yaml(base_config_path)

        # Dynamically point FAISS vector store to tenant specific directory
        tenant_index_path = str(self.local_index_base_dir / db_name)
        
        # Override the vector store config for isolation
        vs_config = VectorStoreConfig(
            provider="faiss",
            index_path=tenant_index_path,
            dimensions=base_config.vector_store.dimensions,
            extra=base_config.vector_store.extra
        )
        base_config.vector_store = vs_config

        return RAGPipeline(base_config)

    def _sync_database(
        self,
        pipeline: RAGPipeline,
        resolved_url: str,
        db_name: str,
        schema_mapping: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Dynamically fetch records from relational database, chunk, embed and store them."""
        # 1. Parse schema mapping or load default
        mapping = schema_mapping or {}
        table_name = mapping.get("table_name", "documents")
        text_columns = mapping.get("text_columns", ["text"])
        metadata_columns = mapping.get("metadata_columns", ["source", "section"])

        logger.info(
            f"Querying database '{db_name}' Table='{table_name}', TextCols={text_columns}, MetaCols={metadata_columns}..."
        )

        # 2. Extract records using SQL database engine
        engine = create_engine(resolved_url)
        
        # Build dynamic select statement
        selected_cols = list(text_columns) + [col for col in metadata_columns if col not in text_columns]
        cols_str = ", ".join(f'"{c}"' for c in selected_cols)
        query_str = f"SELECT {cols_str} FROM {table_name}"
        
        texts_to_ingest: List[Dict[str, Any]] = []

        try:
            with engine.connect() as conn:
                result = conn.execute(text(query_str))
                for row in result:
                    # Convert row to dictionary safely
                    row_dict = row._asdict()
                    
                    # Merge multiple text columns if defined
                    text_parts = [str(row_dict[col]) for col in text_columns if row_dict.get(col) is not None]
                    combined_text = "\n".join(text_parts)
                    
                    if not combined_text.strip():
                        continue

                    # Capture metadata fields
                    meta = {col: row_dict[col] for col in metadata_columns if row_dict.get(col) is not None}
                    meta["db_name"] = db_name
                    meta["table_name"] = table_name

                    texts_to_ingest.append({
                        "text": combined_text,
                        "metadata": meta
                    })
        except Exception as e:
            logger.error(f"Failed to query source database '{db_name}' using SQL query '{query_str}': {e}", exc_info=True)
            raise RuntimeError(f"Database sync failed for database '{db_name}': {e}") from e
        finally:
            engine.dispose()

        # 3. Clear existing vector store index and ingest new texts
        logger.info(f"Clearing old index and ingesting {len(texts_to_ingest)} records for tenant '{db_name}'...")
        pipeline.vector_store.delete(delete_all=True)
        if texts_to_ingest:
            pipeline.ingest_texts(texts_to_ingest)
        else:
            logger.warning(f"No records returned from table '{table_name}' in database '{db_name}'. Vector store is empty.")

    def ask_tenant(
        self,
        db_url: Optional[str],
        db_name: str,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        schema_mapping: Optional[Dict[str, Any]] = None,
        force_sync: bool = False,
        **kwargs,
    ) -> RAGResponse:
        """Dynamic entry point to run an isolated RAG query for a specific database."""
        pipeline = self.get_pipeline(db_url, db_name, schema_mapping=schema_mapping, force_sync=force_sync)
        return pipeline.ask(query, top_k=top_k, filters=filters, **kwargs)

    async def aask_tenant(
        self,
        db_url: Optional[str],
        db_name: str,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        schema_mapping: Optional[Dict[str, Any]] = None,
        force_sync: bool = False,
        **kwargs,
    ) -> RAGResponse:
        """Async dynamic entrypoint to run isolated RAG queries."""
        pipeline = self.get_pipeline(db_url, db_name, schema_mapping=schema_mapping, force_sync=force_sync)
        return await pipeline.aask(query, top_k=top_k, filters=filters, **kwargs)

    # ------------------------------------------------------------------
    # Cache Management Helpers
    # ------------------------------------------------------------------

    def _evict_expired_pipelines(self) -> None:
        """Evict active pipelines that haven't been used in ttl_seconds."""
        now = time.time()
        expired_keys = [
            k for k, entry in self._cache.items()
            if now - entry["last_used"] > self.ttl_seconds
        ]
        for key in expired_keys:
            tenant_name = self._cache[key]["db_name"]
            logger.info(f"LRU Cache: Expiring active RAGPipeline for tenant '{tenant_name}' (TTL exceeded).")
            self._cache.pop(key, None)

    def _evict_lru_if_needed(self) -> None:
        """Evict the oldest pipeline in cache if cache limit exceeded."""
        if len(self._cache) >= self.max_cached_pipelines:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["last_used"])
            tenant_name = self._cache[oldest_key]["db_name"]
            logger.info(f"LRU Cache: Cache limit reached ({self.max_cached_pipelines}). Evicting tenant '{tenant_name}'.")
            self._cache.pop(oldest_key, None)

    def get_stats(self) -> Dict[str, Any]:
        """Retrieve connection pool and cache utilization metrics."""
        self._evict_expired_pipelines()
        active_tenants = [
            {
                "db_name": entry["db_name"],
                "last_active_seconds_ago": int(time.time() - entry["last_used"]),
            }
            for entry in self._cache.values()
        ]
        return {
            "active_connections_count": len(self._cache),
            "max_cache_size": self.max_cached_pipelines,
            "active_tenants": active_tenants,
        }
