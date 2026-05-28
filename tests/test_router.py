"""
tests.test_router
~~~~~~~~~~~~~~~~~
Integration and unit tests verifying the correctness of the dynamic RAGRouter.
Ensures strict database isolation, caching correctness, and dynamic mappings.
"""

import os
import shutil
import sqlite3
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_core.config import RAGConfig
from rag_core.router import RAGRouter


class TestRAGRouter(unittest.TestCase):

    def setUp(self):
        # Set dummy key in environment
        os.environ["OPENAI_API_KEY"] = "mock-testing-key"

        # Temporary files paths
        self.tmp_dir = Path("./tmp_router_test")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        self.db1_path = self.tmp_dir / "bridge.sqlite"
        self.db2_path = self.tmp_dir / "building.sqlite"

        # Create first SQLite database: Bridge Inspections
        conn1 = sqlite3.connect(str(self.db1_path))
        cursor1 = conn1.cursor()
        cursor1.execute(
            """
            CREATE TABLE field_reports (
                id INTEGER PRIMARY KEY,
                details TEXT,
                inspector TEXT,
                site TEXT
            )
            """
        )
        cursor1.execute(
            """
            INSERT INTO field_reports (details, inspector, site)
            VALUES 
            ('INSPECTION NOTES FOR BRIDGE A: Visible cracks in the concrete abutment, repair recommended.', 'John Doe', 'Bridge A'),
            ('INSPECTION NOTES FOR BRIDGE A: Rebar corrosion detected in the underdeck.', 'John Doe', 'Bridge A')
            """
        )
        conn1.commit()
        conn1.close()

        # Create second SQLite database: Building Inspections
        conn2 = sqlite3.connect(str(self.db2_path))
        cursor2 = conn2.cursor()
        cursor2.execute(
            """
            CREATE TABLE building_logs (
                log_id INTEGER PRIMARY KEY,
                notes TEXT,
                auditor TEXT,
                facility_name TEXT
            )
            """
        )
        cursor2.execute(
            """
            INSERT INTO building_logs (notes, auditor, facility_name)
            VALUES 
            ('INSPECTION NOTES FOR BUILDING B: Fire exit door in lobby is blocked.', 'Sarah Connor', 'Building B'),
            ('INSPECTION NOTES FOR BUILDING B: Elevator certificate is expired.', 'Sarah Connor', 'Building B')
            """
        )
        conn2.commit()
        conn2.close()

        # Initialize base config dictionary to use small MiniLM for fast local testing
        self.base_config_dict = {
            "embedder": {
                "provider": "sentence-transformers",
                "model": "all-MiniLM-L6-v2",
            },
            "vector_store": {
                "provider": "faiss",
                "index_path": "./tmp_router_test/indices/default",
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
                "rerank": False,
            },
        }

        # Patch RAGConfig.from_yaml in router so it loads our customized test config dict instead
        self.config_patcher = patch("rag_core.router.RAGConfig.from_yaml")
        self.mock_from_yaml = self.config_patcher.start()
        self.mock_from_yaml.return_value = RAGConfig.from_dict(self.base_config_dict)

        # Instantiate Router
        self.router = RAGRouter()
        self.router.local_index_base_dir = self.tmp_dir / "indices"
        self.router.local_index_base_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.config_patcher.stop()
        shutil.rmtree(str(self.tmp_dir), ignore_errors=True)

    @patch("openai.OpenAI")
    def test_database_isolation_and_routing(self, mock_openai):
        """Verify dynamic RAG routing retrieves data with absolute isolation (tenants never mix)."""
        # Mock LLM generation output
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Mocked LLM generation response."
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        # 1. Query Tenant 1 (Bridge DB)
        bridge_url = f"sqlite:///{self.db1_path.absolute().as_posix()}"
        bridge_mapping = {
            "table_name": "field_reports",
            "text_columns": ["details"],
            "metadata_columns": ["inspector", "site"],
        }
        
        # Retrieve context from Bridge DB
        response_bridge = self.router.ask_tenant(
            db_url=bridge_url,
            db_name="bridge_tenant",
            query="concrete cracks",
            schema_mapping=bridge_mapping,
        )

        # Check tenant 1 retrieval details
        self.assertTrue(len(response_bridge.raw_chunks) > 0)
        for chunk in response_bridge.raw_chunks:
            # Must strictly contain bridge data and bridge metadata
            self.assertIn("BRIDGE A", chunk.text)
            self.assertEqual(chunk.metadata["site"], "Bridge A")
            self.assertEqual(chunk.metadata["db_name"], "bridge_tenant")
            # Must NOT contain building data
            self.assertNotIn("BUILDING B", chunk.text)

        # 2. Query Tenant 2 (Building DB)
        building_url = f"sqlite:///{self.db2_path.absolute().as_posix()}"
        building_mapping = {
            "table_name": "building_logs",
            "text_columns": ["notes"],
            "metadata_columns": ["auditor", "facility_name"],
        }

        # Retrieve context from Building DB
        response_building = self.router.ask_tenant(
            db_url=building_url,
            db_name="building_tenant",
            query="fire door lobby blocked",
            schema_mapping=building_mapping,
        )

        # Check tenant 2 retrieval details
        self.assertTrue(len(response_building.raw_chunks) > 0)
        for chunk in response_building.raw_chunks:
            # Must strictly contain building data and building metadata
            self.assertIn("BUILDING B", chunk.text)
            self.assertEqual(chunk.metadata["facility_name"], "Building B")
            self.assertEqual(chunk.metadata["db_name"], "building_tenant")
            # Must NOT contain bridge data
            self.assertNotIn("BRIDGE A", chunk.text)

    @patch("openai.OpenAI")
    def test_caching_and_lru_eviction(self, mock_openai):
        """Verify connection pool and LRU pipeline cache behavior."""
        # Setup mock OpenAI
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        bridge_url = f"sqlite:///{self.db1_path.absolute().as_posix()}"
        bridge_mapping = {
            "table_name": "field_reports",
            "text_columns": ["details"],
            "metadata_columns": ["inspector", "site"],
        }

        # 1. Pipeline should be added to cache on first call
        self.assertEqual(len(self.router._cache), 0)
        self.router.get_pipeline(
            db_url=bridge_url,
            db_name="bridge_tenant",
            schema_mapping=bridge_mapping,
        )
        self.assertEqual(len(self.router._cache), 1)

        # 2. Subsequent retrieve should hit cache (no new pipeline instantiation)
        pipeline1 = self.router.get_pipeline(
            db_url=bridge_url,
            db_name="bridge_tenant",
            schema_mapping=bridge_mapping,
        )
        pipeline2 = self.router.get_pipeline(
            db_url=bridge_url,
            db_name="bridge_tenant",
            schema_mapping=bridge_mapping,
        )
        self.assertIs(pipeline1, pipeline2)
        self.assertEqual(len(self.router._cache), 1)

        # 3. Test LRU limit
        self.router.max_cached_pipelines = 1
        building_url = f"sqlite:///{self.db2_path.absolute().as_posix()}"
        building_mapping = {
            "table_name": "building_logs",
            "text_columns": ["notes"],
            "metadata_columns": ["auditor", "facility_name"],
        }
        
        # This call should evict bridge_tenant due to max cache size = 1
        self.router.get_pipeline(
            db_url=building_url,
            db_name="building_tenant",
            schema_mapping=building_mapping,
        )
        self.assertEqual(len(self.router._cache), 1)
        # Verify bridge_tenant key is evicted, and building_tenant remains in cache
        active_keys = list(self.router._cache.keys())
        active_entry = self.router._cache[active_keys[0]]
        self.assertEqual(active_entry["db_name"], "building_tenant")


if __name__ == "__main__":
    unittest.main()
