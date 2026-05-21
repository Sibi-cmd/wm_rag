from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class QueryCreate(BaseModel):
    user_id: Optional[int] = None
    vehicle_id: str
    vehicle_model: str
    question: str

class QueryResponse(QueryCreate):
    id: int
    created_at: datetime
    answer: Optional[str] = None

    class Config:
        from_attributes = True

class ServiceHistoryItem(BaseModel):
    serviceId: Optional[int] = None
    serviceDate: Optional[str] = None
    actionTaken: Optional[str] = None       # Spring Boot field name
    resolution: Optional[str] = None        # Spring Boot field name
    # Legacy fields (backward compat)
    serviceType: Optional[str] = None
    description: Optional[str] = None
    providerName: Optional[str] = None

class AIRequest(BaseModel):
    complaintId: Optional[int] = None
    title: Optional[str] = None
    description: Optional[str] = None
    issueType: Optional[str] = None
    priority: Optional[str] = None
    vehicleId: Optional[str] = None
    vehicleModel: Optional[str] = None
    vehicleMake: Optional[str] = None
    yearOfManufacture: Optional[int] = None
    batteryCapacityKwh: Optional[float] = None
    serviceHistory: List[ServiceHistoryItem] = []
    aiAttemptCount: int = 1               # Spring Boot sends 1 for first attempt
    previousSuggestion: Optional[str] = None
    userFollowUp: Optional[str] = None
    userId: Optional[int] = None

class AIResponse(BaseModel):
    suggestion: str
    confidence: float
    predictedCategory: str
    status: Optional[str] = "SUCCESS"    # Spring Boot expects "SUCCESS" or "FALLBACK"

class GatewayQueryCreate(BaseModel):
    userId: Optional[int] = None
    vehicleId: str
    vehicleModel: str
    question: str

class GatewayResponseCreate(BaseModel):
    queryId: int
    userId: Optional[int] = None
    vehicleId: str
    issueId: Optional[str] = None
    answer: str
    confidence: float
    status: str
    title: str
    description: str
