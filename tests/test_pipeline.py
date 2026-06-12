"""
tests.test_pipeline
~~~~~~~~~~~~~~~~~~~
Integration and unit tests verifying the correctness of the WMS RAG pipeline in rag_core.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import rag_core.generators.gemini
import rag_core.vector_stores.qdrant_store
import app.main

from rag_core.config import RAGConfig
from rag_core.pipeline import RAGPipeline


class TestRAGPipeline(unittest.TestCase):

    def setUp(self):
        # Set dummy key in environment to satisfy initialization checks during mock testing
        os.environ["GEMINI_API_KEY"] = "mock-testing-key"

        self.config_dict = {
            "embedder": {
                "provider": "sentence-transformers",
                "model": "all-MiniLM-L6-v2",  # Small and fast model for testing
            },
            "vector_store": {
                "provider": "qdrant",
                "index_name": "test-warehouse-index",
            },
            "generator": {
                "provider": "gemini",
                "model": "gemini-2.5-flash",
            },
            "chunker": {
                "provider": "warehouse",
                "chunk_size": 100,
                "chunk_overlap": 10,
            },
            "retrieval": {
                "top_k": 2,
                "rerank": True,
                "rerank_top_k": 2,
            },
        }
        self.config = RAGConfig.from_dict(self.config_dict)

    @patch("rag_core.vector_stores.qdrant_store.QdrantClient")
    @patch("rag_core.generators.gemini.genai")
    def test_pipeline_creation(self, mock_genai, mock_qdrant):
        """Test that the pipeline initializes and resolves all component factories."""
        pipeline = RAGPipeline(self.config)
        self.assertIsNotNone(pipeline.chunker)
        self.assertIsNotNone(pipeline.embedder)
        self.assertIsNotNone(pipeline.vector_store)
        self.assertIsNotNone(pipeline.generator)
        self.assertIsNotNone(pipeline.reranker)

    @patch("rag_core.vector_stores.qdrant_store.QdrantClient")
    @patch("rag_core.generators.gemini.genai")
    def test_chunker_behavior(self, mock_genai, mock_qdrant):
        """Test that chunks are segmented and have valid identifiers, sections, and priority tags."""
        pipeline = RAGPipeline(self.config)
        text = (
            "PROCEDURE: Reset instructions\n"
            "Disconnect the terminal from the auxiliary system.\n"
            "WARNING: Stay safe from high currents."
        )
        chunks = pipeline.chunker.chunk(text)
        self.assertTrue(len(chunks) > 0)
        for chunk in chunks:
            self.assertIsNotNone(chunk.chunk_id)
            self.assertIn("section", chunk.metadata)
            self.assertIn("severity", chunk.metadata)
            self.assertIn("escalation_required", chunk.metadata)

    @patch("rag_core.vector_stores.qdrant_store.QdrantClient")
    @patch("rag_core.generators.gemini.genai")
    def test_embedder_dimension(self, mock_genai, mock_qdrant):
        """Test that embeddings are generated with matching dimensional weights."""
        pipeline = RAGPipeline(self.config)
        embedding = pipeline.embedder.embed("Verify charging sequence")
        self.assertEqual(len(embedding), 384)

    @patch("rag_core.vector_stores.qdrant_store.QdrantClient")
    @patch("rag_core.generators.gemini.genai")
    def test_end_to_end_flow(self, mock_genai, mock_qdrant):
        """Test document ingestion, retrieval, reranking, and Gemini mock completion loop."""
        mock_client_instance = MagicMock()
        mock_qdrant.return_value = mock_client_instance

        # Mock query points output from Qdrant
        mock_point = MagicMock()
        mock_point.id = "chunk_0_123"
        mock_point.score = 0.9
        mock_point.payload = {
            "metadata": {
                "source": "manual.pdf",
                "section": "Reset",
                "severity": "medium",
                "text": "Disconnect the terminal from the auxiliary system."
            }
        }
        
        mock_results = MagicMock()
        mock_results.points = [mock_point]
        mock_client_instance.query_points.return_value = mock_results

        # Mock Gemini Client response (google-genai SDK)
        mock_response = MagicMock()
        mock_response.text = "Procedure completed successfully."
        
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        pipeline = RAGPipeline(self.config)

        # Retrieve and verify database queries
        results = pipeline.retrieve("Reset instructions", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["source"], "manual.pdf")

        # Query pipeline Q&A
        response = pipeline.ask("How to perform reset?")
        self.assertEqual(response.text, "Procedure completed successfully.")
        self.assertIn("manual.pdf", response.sources)
        self.assertTrue(response.confidence > 0.0)

    @patch("rag_core.vector_stores.qdrant_store.QdrantClient")
    @patch("rag_core.generators.gemini.genai")
    def test_config_defaults(self, mock_genai, mock_qdrant):
        """Test that default RAGConfig uses the correct providers."""
        default_config = RAGConfig()
        self.assertEqual(default_config.vector_store.provider, "qdrant")
        self.assertEqual(default_config.generator.provider, "gemini")
        self.assertEqual(default_config.generator.model, "gemini-2.5-flash")
        self.assertEqual(default_config.generator.api_key_env, "GEMINI_API_KEY")
        self.assertEqual(default_config.chunker.provider, "warehouse")

    def test_chunker_config_provider(self):
        """Test that ChunkerConfig correctly parses the provider field."""
        config = RAGConfig.from_dict(self.config_dict)
        self.assertEqual(config.chunker.provider, "warehouse")

    def test_ingest_request_validation(self):
        """Test IngestRequest schema validation with full metadata."""
        from app.models import IngestRequest
        payload = {
            "ocr_document_id": "ocr-doc-123",
            "document_type": "StockReport",
            "warehouse_id": "WH001",
            "text": "Product SKU100 is in Zone C.",
            "sku": "SKU100",
            "product_id": "PROD-20",
            "category": "Electronics",
            "zone": "Zone C",
            "rack": "R1",
            "shelf": "S2",
            "bin": "B4"
        }
        req = IngestRequest(**payload)
        self.assertEqual(req.ocr_document_id, "ocr-doc-123")
        self.assertEqual(req.sku, "SKU100")
        self.assertEqual(req.shelf, "S2")

    def test_ingest_request_backward_compatibility(self):
        """Test IngestRequest schema validation defaults to None for missing fields (backward compatibility)."""
        from app.models import IngestRequest
        payload = {
            "ocr_document_id": "ocr-doc-123",
            "document_type": "StockReport",
            "warehouse_id": "WH001",
            "text": "Product SKU100 is in Zone C."
        }
        req = IngestRequest(**payload)
        self.assertEqual(req.ocr_document_id, "ocr-doc-123")
        self.assertIsNone(req.sku)
        self.assertIsNone(req.shelf)

    @patch("app.main.mongo_db")
    @patch("rag_core.vector_stores.qdrant_store.QdrantClient")
    @patch("rag_core.generators.gemini.genai")
    @patch("app.main.rag_pipeline")
    def test_api_ingest_endpoint(self, mock_rag_pipeline, mock_genai, mock_qdrant, mock_mongo):
        """Test the rag_ingest endpoint and verify Qdrant points persistence payload structure."""
        from app.main import rag_ingest
        from app.models import IngestRequest
        
        # Setup mock pipeline
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.core_pipeline.embedder.embed.return_value = [0.1] * 768
        mock_rag_pipeline.core_pipeline = mock_pipeline_instance.core_pipeline
        
        # Mock Qdrant Client return
        mock_qdrant_instance = MagicMock()
        mock_pipeline_instance.core_pipeline.vector_store._client = mock_qdrant_instance
        
        payload = {
            "ocr_document_id": "ocr-doc-123",
            "document_type": "StockReport",
            "warehouse_id": "WH001",
            "text": "Product SKU100 is in Zone C.",
            "sku": "SKU100",
            "product_id": "PROD-20",
            "category": "Electronics",
            "zone": "Zone C",
            "rack": "R1",
            "shelf": "S2",
            "bin": "B4"
        }
        req = IngestRequest(**payload)
        
        # Run async handler
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            res = loop.run_until_complete(rag_ingest(req))
        finally:
            loop.close()
        
        self.assertEqual(res["status"], "SUCCESS")
        
        # Verify upsert call
        self.assertTrue(mock_qdrant_instance.upsert.called)
        called_args = mock_qdrant_instance.upsert.call_args
        points = called_args[1]["points"]
        self.assertTrue(len(points) > 0)
        
        # Verify Qdrant payload validation (metadata nested & flat structure)
        payload = points[0].payload
        self.assertIn("metadata", payload)
        self.assertEqual(payload["sku"], "SKU100")
        self.assertEqual(payload["shelf"], "S2")
        self.assertEqual(payload["metadata"]["sku"], "SKU100")
        self.assertEqual(payload["metadata"]["shelf"], "S2")

    def test_build_qdrant_filter(self):
        """Test build_qdrant_filter creates correct FieldConditions and prefixes keys."""
        from rag_core.vector_stores.qdrant_store import QdrantStore
        
        filters = {
            "sku": "SKU100",
            "metadata.zone": "Zone A",
            "shelf": None
        }
        q_filter = QdrantStore.build_qdrant_filter(filters)
        self.assertIsNotNone(q_filter)
        conditions = q_filter.must
        self.assertEqual(len(conditions), 2)
        
        # Verify keys are correctly prefixed and None is ignored
        keys = [c.key for c in conditions]
        self.assertIn("metadata.sku", keys)
        self.assertIn("metadata.zone", keys)
        self.assertNotIn("metadata.shelf", keys)

    @patch("rag_core.vector_stores.qdrant_store.QdrantClient")
    @patch("rag_core.generators.gemini.genai")
    def test_sku_filtering(self, mock_genai, mock_qdrant):
        """Verify SKU filter is propagated to Qdrant search."""
        mock_client_instance = MagicMock()
        mock_qdrant.return_value = mock_client_instance
        
        pipeline = RAGPipeline(self.config)
        
        # Trigger retrieve with filters
        pipeline.retrieve("Reset instructions", filters={"sku": "SKU100"})
        
        self.assertTrue(mock_client_instance.query_points.called)
        called_args = mock_client_instance.query_points.call_args
        query_filter = called_args[1]["query_filter"]
        self.assertIsNotNone(query_filter)
        
        # Verify conditions
        self.assertEqual(len(query_filter.must), 1)
        self.assertEqual(query_filter.must[0].key, "metadata.sku")
        self.assertEqual(query_filter.must[0].match.value, "SKU100")

    @patch("rag_core.vector_stores.qdrant_store.QdrantClient")
    @patch("rag_core.generators.gemini.genai")
    def test_location_filtering(self, mock_genai, mock_qdrant):
        """Verify location filters (zone, rack, shelf, bin) are structured in generated filter."""
        mock_client_instance = MagicMock()
        mock_qdrant.return_value = mock_client_instance
        
        pipeline = RAGPipeline(self.config)
        
        location_filters = {
            "zone": "Zone A",
            "rack": "Rack 1",
            "shelf": "Shelf 2",
            "bin": "Bin 3"
        }
        pipeline.retrieve("Locate inventory", filters=location_filters)
        
        self.assertTrue(mock_client_instance.query_points.called)
        query_filter = mock_client_instance.query_points.call_args[1]["query_filter"]
        self.assertIsNotNone(query_filter)
        
        keys = [c.key for c in query_filter.must]
        self.assertEqual(len(keys), 4)
        self.assertIn("metadata.zone", keys)
        self.assertIn("metadata.rack", keys)
        self.assertIn("metadata.shelf", keys)
        self.assertIn("metadata.bin", keys)

    @patch("rag_core.vector_stores.qdrant_store.QdrantClient")
    @patch("rag_core.generators.gemini.genai")
    def test_compound_filtering(self, mock_genai, mock_qdrant):
        """Verify compound filtering logic combines multiple attributes using must (AND)."""
        mock_client_instance = MagicMock()
        mock_qdrant.return_value = mock_client_instance
        
        pipeline = RAGPipeline(self.config)
        
        compound_filters = {
            "sku": "SKU200",
            "zone": "Zone B",
            "bin": "Bin 5"
        }
        pipeline.retrieve("Verify item", filters=compound_filters)
        
        query_filter = mock_client_instance.query_points.call_args[1]["query_filter"]
        self.assertIsNotNone(query_filter)
        
        # Verify they are grouped under must list (representing AND)
        self.assertEqual(len(query_filter.must), 3)
        keys = [c.key for c in query_filter.must]
        self.assertIn("metadata.sku", keys)
        self.assertIn("metadata.zone", keys)
        self.assertIn("metadata.bin", keys)

    @patch("rag_core.vector_stores.qdrant_store.QdrantClient")
    @patch("rag_core.generators.gemini.genai")
    def test_filtering_backward_compatibility(self, mock_genai, mock_qdrant):
        """Verify search executes without query_filter when filters dictionary is empty or None."""
        mock_client_instance = MagicMock()
        mock_qdrant.return_value = mock_client_instance
        
        pipeline = RAGPipeline(self.config)
        
        # Call retrieve with empty filters
        pipeline.retrieve("Locate stock", filters={})
        query_filter = mock_client_instance.query_points.call_args[1]["query_filter"]
        self.assertIsNone(query_filter)
        
        # Call retrieve with None
        pipeline.retrieve("Locate stock", filters=None)
        query_filter = mock_client_instance.query_points.call_args[1]["query_filter"]
        self.assertIsNone(query_filter)


if __name__ == "__main__":
    unittest.main()
