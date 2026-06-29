from sqlalchemy import Column, String, Text, DateTime, Float, Integer
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class OCRDocumentModel(Base):
    __tablename__ = 'ocr_documents'

    id = Column(String(36), primary_key=True, name='ocr_id')
    file_name = Column(String(255))
    file_path = Column(String(512))
    document_type = Column(String(100))
    document_hash = Column(String(64), nullable=True)
    raw_text = Column(Text, default='')
    extracted_json = Column(Text, nullable=True)
    confidence_score = Column(Float, nullable=True)
    processing_status = Column(String(50), default='UPLOADED')
    error_message = Column(Text, nullable=True)
    rejection_reason = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, nullable=True)

    # WMS Metadata fields for RAG
    warehouse_id = Column(String(100), nullable=True)
    sku = Column(String(100), nullable=True)
    product_id = Column(String(100), nullable=True)
    category = Column(String(100), nullable=True)
    zone = Column(String(100), nullable=True)
    rack = Column(String(100), nullable=True)
    shelf = Column(String(100), nullable=True)
    bin = Column(String(100), nullable=True)
    chunk_count = Column(Integer, default=0)
    rag_status = Column(String(50), default='PENDING')
    rag_error_message = Column(Text, nullable=True)
