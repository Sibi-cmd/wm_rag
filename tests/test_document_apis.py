import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

# Resolve Python 3.14 patch imports beforehand
import app.main
from app.main import app

client = TestClient(app)


class TestDocumentAPIs:
    @patch("app.main.get_db_session")
    def test_list_documents_success(self, mock_get_db):
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        # Setup mock db query output
        mock_doc = MagicMock()
        mock_doc.id = "doc-999"
        mock_doc.document_type = "Invoice"
        mock_doc.warehouse_id = "WH001"
        mock_doc.sku = "SKU-999"
        mock_doc.product_id = "prod-999"
        mock_doc.category = "Tools"
        mock_doc.zone = "Zone A"
        mock_doc.rack = "Rack 1"
        mock_doc.shelf = "1"
        mock_doc.bin = "Bin 1"
        mock_doc.chunk_count = 3
        mock_doc.created_at = datetime(2026, 6, 12, 12, 0, 0, tzinfo=timezone.utc)
        mock_doc.updated_at = datetime(2026, 6, 12, 12, 0, 0, tzinfo=timezone.utc)

        mock_db.query.return_value.filter.return_value.all.return_value = [mock_doc]

        response = client.get("/api/rag/documents")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["ocr_document_id"] == "doc-999"
        assert data[0]["chunk_count"] == 3
        assert data[0]["sku"] == "SKU-999"

    @patch("app.main.get_qdrant_client")
    @patch("app.main.get_db_session")
    def test_get_document_details_success(self, mock_get_db, mock_get_qdrant):
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        # Setup mock document in db
        mock_doc = MagicMock()
        mock_doc.id = "doc-999"
        mock_doc.document_type = "Invoice"
        mock_doc.warehouse_id = "WH001"
        mock_doc.sku = "SKU-999"
        mock_doc.product_id = "prod-999"
        mock_doc.category = "Tools"
        mock_doc.zone = "Zone A"
        mock_doc.rack = "Rack 1"
        mock_doc.shelf = "1"
        mock_doc.bin = "Bin 1"
        mock_doc.chunk_count = 2
        mock_db.query.return_value.filter.return_value.first.return_value = mock_doc

        # Setup mock scroll results from Qdrant
        mock_qdrant = MagicMock()
        mock_get_qdrant.return_value = mock_qdrant

        mock_point1 = MagicMock()
        mock_point1.id = "chunk-1"
        mock_point1.payload = {
            "metadata": {
                "chunk_id": "doc-999_0",
                "chunk_index": 0,
                "text": "First chunk text details"
            }
        }

        mock_point2 = MagicMock()
        mock_point2.id = "chunk-2"
        mock_point2.payload = {
            "metadata": {
                "chunk_id": "doc-999_1",
                "chunk_index": 1,
                "text": "Second chunk text details"
            }
        }

        mock_qdrant.scroll.return_value = ([mock_point1, mock_point2], None)

        response = client.get("/api/rag/documents/doc-999")
        assert response.status_code == 200
        data = response.json()
        assert data["ocr_document_id"] == "doc-999"
        assert len(data["chunks"]) == 2
        assert data["chunks"][0]["chunk_id"] == "doc-999_0"
        assert data["chunks"][0]["text"] == "First chunk text details"

    @patch("app.main.get_db_session")
    def test_get_document_details_not_found(self, mock_get_db):
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_db.query.return_value.filter.return_value.first.return_value = None

        response = client.get("/api/rag/documents/doc-missing")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @patch("app.main.get_qdrant_client")
    @patch("app.main.get_db_session")
    def test_delete_document_success(self, mock_get_db, mock_get_qdrant):
        # Mock Qdrant
        mock_qdrant = MagicMock()
        mock_get_qdrant.return_value = mock_qdrant
        mock_point = MagicMock()
        mock_qdrant.scroll.return_value = ([mock_point] * 5, None)

        # Mock DB
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_doc = MagicMock()
        mock_doc.chunk_count = 5
        mock_db.query.return_value.filter.return_value.first.return_value = mock_doc

        response = client.delete("/api/rag/documents/doc-999")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SUCCESS"
        assert data["deleted_count"] == 5

        # Verify Qdrant and DB calls
        mock_qdrant.delete.assert_called_once()
        mock_db.commit.assert_called_once()

    @patch("app.main.get_qdrant_client")
    @patch("app.main.get_db_session")
    def test_get_collections_stats_success(self, mock_get_db, mock_get_qdrant):
        # Mock DB
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_db.query.return_value.filter.return_value.all.return_value = [MagicMock(), MagicMock()]  # 2 docs
        mock_db.query.return_value.filter.return_value.scalar.return_value = 10  # 10 chunks total

        # Mock Qdrant
        mock_qdrant = MagicMock()
        mock_get_qdrant.return_value = mock_qdrant
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 10
        mock_qdrant.get_collection.return_value = mock_collection_info

        response = client.get("/api/rag/collections/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["collection_name"] == "warehouse-index"
        assert data["total_documents"] == 2
        assert data["total_chunks"] == 10
        assert data["vector_count"] == 10
