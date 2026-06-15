from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class QueryCreate(BaseModel):
    user_id: Optional[int] = None
    ocr_document_id: str
    document_type: str
    question: str

class QueryResponse(QueryCreate):
    id: int
    created_at: datetime
    answer: Optional[str] = None

    class Config:
        from_attributes = True

class AuditItem(BaseModel):
    auditId: Optional[int] = None
    auditDate: Optional[str] = None
    actionTaken: Optional[str] = None
    resolution: Optional[str] = None
    description: Optional[str] = None
    sourceSystem: Optional[str] = None

class WarehouseRequest(BaseModel):
    ocrDocumentId: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    documentType: Optional[str] = None
    document_type: Optional[str] = None
    priority: Optional[str] = None
    warehouseId: Optional[str] = None
    warehouse_id: Optional[str] = None
    sku: Optional[str] = None
    productId: Optional[str] = None
    product_id: Optional[str] = None
    category: Optional[str] = None
    zone: Optional[str] = None
    rack: Optional[str] = None
    shelf: Optional[str] = None
    bin: Optional[str] = None
    auditTrail: List[AuditItem] = []
    attemptCount: int = 1
    previousResponse: Optional[str] = None
    followUpQuestion: Optional[str] = None
    userId: Optional[int] = None

class WarehouseResponse(BaseModel):
    suggestion: str
    confidence: float
    predictedCategory: str
    status: Optional[str] = "SUCCESS"

class WarehouseGatewayQuery(BaseModel):
    userId: Optional[int] = None
    warehouseId: str
    documentType: str
    question: str

class WarehouseGatewayResponse(BaseModel):
    queryId: int
    userId: Optional[int] = None
    warehouseId: str
    ocrDocumentId: Optional[str] = None
    answer: str
    confidence: float
    status: str
    title: str
    description: str

class IngestRequest(BaseModel):
    ocr_document_id: str
    document_type: str
    warehouse_id: str
    text: str
    sku: Optional[str] = None
    product_id: Optional[str] = None
    category: Optional[str] = None
    zone: Optional[str] = None
    rack: Optional[str] = None
    shelf: Optional[str] = None
    bin: Optional[str] = None


class DocumentListItem(BaseModel):
    document_id: str
    document_type: Optional[str] = None
    warehouse_id: Optional[str] = None
    sku: Optional[str] = None
    zone: Optional[str] = None
    rack: Optional[str] = None
    shelf: Optional[str] = None
    bin: Optional[str] = None
    chunk_count: int


class DocumentMetadata(BaseModel):
    document_id: str
    document_type: Optional[str] = None
    warehouse_id: Optional[str] = None


class ChunkDetail(BaseModel):
    chunk_index: int
    text: str


class DocumentDetailResponse(BaseModel):
    metadata: DocumentMetadata
    chunks: List[ChunkDetail]


class DocumentDeleteResponse(BaseModel):
    status: str = "SUCCESS"
    document_id: str


class CollectionStatsResponse(BaseModel):
    collection_name: str
    total_documents: int
    total_chunks: int
    vector_count: int


