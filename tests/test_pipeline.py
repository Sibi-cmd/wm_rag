"""
tests.test_pipeline
~~~~~~~~~~~~~~~~~~~
Integration and unit tests verifying the correctness of the rag_core library.
"""

import os
import shutil
import unittest
from unittest.mock import MagicMock, patch

from rag_core.config import RAGConfig
from rag_core.pipeline import RAGPipeline


class TestRAGPipeline(unittest.TestCase):

    def setUp(self):
        # Set dummy key in environment to satisfy initialization checks during mock testing
        os.environ["OPENAI_API_KEY"] = "mock-testing-key"

        self.config_dict = {
            "embedder": {
                "provider": "sentence-transformers",
                "model": "all-MiniLM-L6-v2",  # Small and fast model for testing
            },
            "vector_store": {
                "provider": "faiss",
                "index_path": "./tmp_faiss_test",
                "dimensions": 384,
            },
            "generator": {
                "provider": "openai",
                "model": "gpt-4o-mini",
            },
            "chunker": {
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

    def tearDown(self):
        shutil.rmtree("./tmp_faiss_test", ignore_errors=True)

    def test_pipeline_creation(self):
        """Test that the pipeline initializes and resolves all component factories."""
        pipeline = RAGPipeline(self.config)
        self.assertIsNotNone(pipeline.chunker)
        self.assertIsNotNone(pipeline.embedder)
        self.assertIsNotNone(pipeline.vector_store)
        self.assertIsNotNone(pipeline.generator)
        self.assertIsNotNone(pipeline.reranker)

    def test_chunker_behavior(self):
        """Test that chunks are segmented and have valid identifiers."""
        pipeline = RAGPipeline(self.config)
        text = (
            "PROCEDURE: Reset instructions\n"
            "Disconnect the terminal from the auxiliary system.\n"
            "WARNING: Stay safe from high currents."
        )
        chunks = pipeline.chunker.chunk(text)
        self.assertTrue(len(chunks) > 0)
        for chunk in chunks:
            self.assertTrue(chunk.chunk_id.startswith("chunk_"))
            self.assertIn("section", chunk.metadata)

    def test_embedder_dimension(self):
        """Test that embeddings are generated with matching dimensional weights."""
        pipeline = RAGPipeline(self.config)
        embedding = pipeline.embedder.embed("Verify charging sequence")
        self.assertEqual(len(embedding), 384)

    @patch("openai.OpenAI")
    def test_end_to_end_flow(self, mock_openai):
        """Test document ingestion, retrieval, keyword reranking, and OpenAI mock completion loop."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Procedure completed successfully."
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        pipeline = RAGPipeline(self.config)
        pipeline.ingest_texts([
            {
                "text": "PROCEDURE: Reset instructions. Disconnect the terminal from the auxiliary system.",
                "metadata": {"source": "manual.pdf"}
            }
        ])

        # Retrieve and verify database queries
        results = pipeline.retrieve("Reset instructions", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["source"], "manual.pdf")

        # Query pipeline Q&A
        response = pipeline.ask("How to perform reset?")
        self.assertEqual(response.text, "Procedure completed successfully.")
        self.assertIn("manual.pdf", response.sources)
        self.assertTrue(response.confidence > 0.0)


if __name__ == "__main__":
    unittest.main()
