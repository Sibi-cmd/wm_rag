"""
tests.test_pipeline
~~~~~~~~~~~~~~~~~~~
Integration and unit tests verifying the correctness of the WMS RAG pipeline in rag_core.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

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


if __name__ == "__main__":
    unittest.main()
