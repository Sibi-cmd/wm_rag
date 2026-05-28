"""
rag_core.pipeline
~~~~~~~~~~~~~~~~~
Main orchestrator for the RAG pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .config import RAGConfig
from .types import Chunk, Document, RAGResponse, SearchResult

logger = logging.getLogger("rag_core.pipeline")


class RAGPipeline:
    """The main entry point for the rag_core library.

    Orchestrates the ingestion, embedding, storage, retrieval, reranking,
    and LLM generation steps.

    Args:
        config: A populated ``RAGConfig`` instance.
    """

    def __init__(self, config: RAGConfig) -> None:
        self.config = config

        # Import registries and create components
        from .chunkers import create_chunker
        from .embedders import create_embedder
        from .generators import create_generator
        from .rerankers import create_reranker
        from .vector_stores import create_vector_store

        logger.info("Initializing RAGPipeline components...")
        self.chunker = create_chunker(config.chunker)
        self.embedder = create_embedder(config.embedder)
        self.vector_store = create_vector_store(config.vector_store)
        self.generator = create_generator(config.generator)

        # Reranker is optional but default is enabled
        self.reranker = None
        if config.retrieval.rerank:
            reranker_provider = config.retrieval.extra.get("reranker_provider", "keyword")
            self.reranker = create_reranker(reranker_provider)

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> RAGPipeline:
        """Create a pipeline from a YAML configuration file.

        Args:
            config_path: Path to the YAML file.

        Returns:
            An instantiated RAGPipeline.
        """
        config = RAGConfig.from_yaml(config_path)
        return cls(config)

    # ------------------------------------------------------------------
    # Ingestion APIs
    # ------------------------------------------------------------------

    def ingest_files(self, file_paths: List[Union[str, Path]]) -> None:
        """Read, chunk, embed, and store files.

        Supports PDF, TXT, and MD formats.

        Args:
            file_paths: A list of local file paths.
        """
        all_chunks: List[Chunk] = []
        for path in file_paths:
            path_str = str(path)
            logger.info(f"Ingesting file: {path_str}")
            try:
                chunks = self.chunker.chunk_file(path_str)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to ingest file '{path_str}': {e}", exc_info=True)
                raise RuntimeError(f"Failed to ingest file '{path_str}': {e}") from e

        if not all_chunks:
            logger.warning("No chunks generated from the provided files.")
            return

        self._embed_and_store(all_chunks)

    def ingest_texts(self, texts: List[Dict[str, Any]]) -> None:
        """Ingest raw texts directly with metadata.

        Args:
            texts: A list of dicts. Each dict must have a ``"text"`` key
                   and an optional ``"metadata"`` key (dict).
                   Example:
                       [{"text": "my doc text", "metadata": {"source": "manual.txt"}}]
        """
        all_chunks: List[Chunk] = []
        for idx, item in enumerate(texts):
            text = item.get("text", "")
            if not text.strip():
                continue
            meta = item.get("metadata", {})
            chunks = self.chunker.chunk(text, metadata=meta)
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("No chunks generated from the provided texts.")
            return

        self._embed_and_store(all_chunks)

    def _embed_and_store(self, chunks: List[Chunk]) -> None:
        """Internal helper to embed chunks and upsert to vector store."""
        logger.info(f"Embedding {len(chunks)} chunks...")
        texts_to_embed = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed_batch(texts_to_embed)

        vectors_to_upsert = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            # Add text directly to metadata so vector stores can return it during query
            meta = {**(chunk.metadata or {})}
            meta["text"] = chunk.text
            vectors_to_upsert.append(
                {
                    "id": chunk.chunk_id,
                    "values": embedding,
                    "metadata": meta,
                }
            )

        logger.info(f"Storing vectors in the vector store ({self.config.vector_store.provider})...")
        self.vector_store.upsert(vectors_to_upsert)
        logger.info("Ingestion completed successfully.")

    # ------------------------------------------------------------------
    # Query & Retrieval APIs
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Find relevant chunks for a query.

        Args:
            query: The user search query.
            top_k: Optional override for retrieval top_k.
            filters: Optional metadata filters.

        Returns:
            A list of ``SearchResult`` objects.
        """
        # 1. Embed query
        query_vector = self.embedder.embed(query)

        # 2. Retrieve initial candidates
        search_top_k = top_k or self.config.retrieval.top_k
        # If reranking is enabled, retrieve more candidates first
        if self.config.retrieval.rerank:
            initial_k = max(search_top_k * 3, 10)
        else:
            initial_k = search_top_k

        candidates = self.vector_store.search(
            vector=query_vector,
            top_k=initial_k,
            filters=filters,
        )

        if not candidates:
            return []

        # 3. Rerank if configured
        if self.config.retrieval.rerank and self.reranker:
            final_k = top_k or self.config.retrieval.rerank_top_k
            return self.reranker.rerank(query, candidates, top_k=final_k)

        return candidates[:search_top_k]

    def generate(self, prompt: str, **kwargs) -> str:
        """Call the LLM generator directly with a custom prompt."""
        return self.generator.generate(prompt, **kwargs)

    async def agenerate(self, prompt: str, **kwargs) -> str:
        """Call the LLM generator asynchronously with a custom prompt."""
        return await self.generator.agenerate(prompt, **kwargs)

    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> RAGResponse:
        """Ask a question and get a generated answer with sources.

        Args:
            question: The user question.
            top_k: Optional override for top_k retrieved chunks.
            filters: Optional metadata filters.
            **kwargs: Extra LLM generation parameter overrides.

        Returns:
            A ``RAGResponse`` object.
        """
        # 1. Retrieve context
        results = self.retrieve(question, top_k=top_k, filters=filters)

        if not results:
            return RAGResponse(
                text="I don't have enough information to answer this question.",
                sources=[],
                confidence=0.0,
                chunks_used=0,
                raw_chunks=[],
            )

        # 2. Format context
        context_str = "\n\n".join(
            f"[Source: {r.metadata.get('source', 'unknown')} | Section: {r.metadata.get('section', 'General')}]\n{r.text}"
            for r in results
        )

        # 3. Render prompt template
        prompt = self.config.prompt_template.format(
            context=context_str,
            question=question,
        )

        # 4. Generate answer
        answer_text = self.generate(prompt, **kwargs)

        # 5. Extract sources
        sources = []
        for r in results:
            src = r.metadata.get("source")
            if src and src not in sources:
                sources.append(src)

        # 6. Calculate confidence score (highest similarity score of the top chunk)
        confidence = results[0].score if results else 0.0

        return RAGResponse(
            text=answer_text,
            sources=sources,
            confidence=confidence,
            chunks_used=len(results),
            raw_chunks=results,
        )

    async def aask(
        self,
        question: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> RAGResponse:
        """Async version of ``ask()``."""
        # 1. Retrieve context (blocking call as vector stores are typically sync)
        results = self.retrieve(question, top_k=top_k, filters=filters)

        if not results:
            return RAGResponse(
                text="I don't have enough information to answer this question.",
                sources=[],
                confidence=0.0,
                chunks_used=0,
                raw_chunks=[],
            )

        # 2. Format context
        context_str = "\n\n".join(
            f"[Source: {r.metadata.get('source', 'unknown')} | Section: {r.metadata.get('section', 'General')}]\n{r.text}"
            for r in results
        )

        # 3. Render prompt template
        prompt = self.config.prompt_template.format(
            context=context_str,
            question=question,
        )

        # 4. Generate answer asynchronously
        answer_text = await self.agenerate(prompt, **kwargs)

        # 5. Extract sources
        sources = []
        for r in results:
            src = r.metadata.get("source")
            if src and src not in sources:
                sources.append(src)

        confidence = results[0].score if results else 0.0

        return RAGResponse(
            text=answer_text,
            sources=sources,
            confidence=confidence,
            chunks_used=len(results),
            raw_chunks=results,
        )
