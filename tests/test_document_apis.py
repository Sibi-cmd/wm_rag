import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

# Resolve Python 3.14 patch imports beforehand
import app.main
from app.main import app

client = TestClient(app)


class TestDocumentAPIs:
    @patch("app.main.mongo_db")
    def test_list_documents_success(self, mock_mongo):
        # Setup mock db aggregate output
        mock_mongo.chunks.aggregate.return_value = [
            {
                "_id": "doc-999",
                "chunk_count": 3,
                "document_type": "Invoice",
                "warehouse_id": "WH001",
                "sku": "SKU-999",
                "product_id": "prod-999",
                "category": "Tools",
                "zone": "Zone A",
                "rack": "Rack 1",
                "shelf": "1",
                "bin": "Bin 1",
                "updated_at": datetime(2026, 6, 12, 12, 0, 0, tzinfo=timezone.utc)
            }
        ]

        response = client.get("/api/rag/documents")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["ocr_document_id"] == "doc-999"
        assert data[0]["chunk_count"] == 3
        assert data[0]["sku"] == "SKU-999"

    @patch("app.main.mongo_db")
    def test_get_document_details_success(self, mock_mongo):
        # Setup mock cursor
        mock_mongo.chunks.find.return_value.sort.return_value = [
            {
                "chunk_id": "doc-999_0",
                "ocr_document_id": "doc-999",
                "chunk_index": 0,
                "text": "First chunk text details",
                "document_type": "Invoice",
                "sku": "SKU-999",
            },
            {
                "chunk_id": "doc-999_1",
                "ocr_document_id": "doc-999",
                "chunk_index": 1,
                "text": "Second chunk text details",
                "document_type": "Invoice",
                "sku": "SKU-999",
            }
        ]

        response = client.get("/api/rag/documents/doc-999")
        assert response.status_code == 200
        data = response.json()
        assert data["ocr_document_id"] == "doc-999"
        assert len(data["chunks"]) == 2
        assert data["chunks"][0]["chunk_id"] == "doc-999_0"
        assert data["chunks"][0]["text"] == "First chunk text details"

    @patch("app.main.mongo_db")
    def test_get_document_details_not_found(self, mock_mongo):
        mock_mongo.chunks.find.return_value.sort.return_value = []

        response = client.get("/api/rag/documents/doc-missing")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @patch("app.main.get_qdrant_client")
    @patch("app.main.mongo_db")
    def test_delete_document_success(self, mock_mongo, mock_get_qdrant):
        # Mock Qdrant delete
        mock_qdrant = MagicMock()
        mock_get_qdrant.return_value = mock_qdrant

        # Mock MongoDB chunks delete_many
        mock_mongo.chunks.delete_many.return_value.deleted_count = 5

        response = client.delete("/api/rag/documents/doc-999")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SUCCESS"
        assert data["deleted_count"] == 5

        # Verify Qdrant and Mongo delete calls
        mock_qdrant.delete.assert_called_once()
        mock_mongo.chunks.delete_many.assert_called_once_with({"ocr_document_id": "doc-999"})

    @patch("app.main.get_qdrant_client")
    @patch("app.main.mongo_db")
    def test_get_collections_stats_success(self, mock_mongo, mock_get_qdrant):
        mock_qdrant = MagicMock()
        mock_get_qdrant.return_value = mock_qdrant

        # Mock unique documents count & chunks count
        mock_mongo.chunks.distinct.return_value = ["doc-1", "doc-2"]
        mock_mongo.chunks.count_documents.return_value = 10

        # Mock Qdrant collection points info
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
