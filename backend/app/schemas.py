from datetime import date, datetime, timezone
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

# --- AI COPILOT SCHEMAS ---

class CompletenessFlag(BaseModel):
    field: str
    status: str = "missing"  # 'missing', 'ambiguous', 'complete'
    note: str

class DuplicateWarning(BaseModel):
    is_duplicate: bool = False
    matched_complaint_id: Optional[str] = None
    matched_complaint_number: Optional[str] = None
    confidence: float = 0.0
    details: Optional[str] = None

class ExtractedEntities(BaseModel):
    customer_name: Optional[str] = None
    customer_contact: Optional[str] = None
    product_name: Optional[str] = None
    product_type: Optional[str] = None  # 'API', 'FDF'
    dosage_form: Optional[str] = None
    batch_number: Optional[str] = None
    manufacture_date: Optional[str] = None
    expiry_date: Optional[str] = None
    complaint_date: Optional[str] = None
    complaint_category: Optional[str] = None
    complaint_description: Optional[str] = None

class AIAnalysisResult(BaseModel):
    extracted_fields: ExtractedEntities
    completeness_flags: List[CompletenessFlag] = []
    duplicate_warning: DuplicateWarning = Field(default_factory=DuplicateWarning)
    risk_level: str = "MINOR"  # 'CRITICAL', 'MAJOR', 'MINOR'
    risk_rationale: str
    capa_suggestion: str
    summary: str
    model_version: str = "Groq Llama-3.1-8B / Llama-3.3-70B"
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AnalyzeTextRequest(BaseModel):
    raw_text: str
    source_type: str = "TEXT"  # 'TEXT', 'EMAIL'

# --- COMPLAINT CRUD SCHEMAS ---

class ComplaintBase(BaseModel):
    customer_name: Optional[str] = None
    customer_contact: Optional[str] = None
    product_name: str
    product_type: Optional[str] = "FDF"
    dosage_form: Optional[str] = None
    batch_number: str
    manufacture_date: Optional[date] = None
    expiry_date: Optional[date] = None
    complaint_date: Optional[date] = None
    complaint_category: str
    complaint_description: str
    severity_classification: str = "MINOR"
    qa_reviewer_notes: Optional[str] = None
    capa_required: bool = False
    capa_action_plan: Optional[str] = None
    assigned_investigator: Optional[str] = None
    status: str = "LOGGED"

class ComplaintCreate(ComplaintBase):
    raw_source_type: str = "TEXT"
    raw_text: str
    raw_file_name: Optional[str] = None
    ai_extracted_json: Optional[Dict[str, Any]] = None
    ai_risk_level: Optional[str] = None
    ai_risk_rationale: Optional[str] = None
    ai_capa_suggestion: Optional[str] = None
    ai_summary: Optional[str] = None
    ai_completeness_flags: Optional[List[Dict[str, Any]]] = None
    ai_duplicate_warning: Optional[Dict[str, Any]] = None
    confirmed_by_user: Optional[str] = "QA Officer"

class ComplaintUpdate(BaseModel):
    customer_name: Optional[str] = None
    customer_contact: Optional[str] = None
    product_name: Optional[str] = None
    product_type: Optional[str] = None
    dosage_form: Optional[str] = None
    batch_number: Optional[str] = None
    manufacture_date: Optional[date] = None
    expiry_date: Optional[date] = None
    complaint_date: Optional[date] = None
    complaint_category: Optional[str] = None
    complaint_description: Optional[str] = None
    severity_classification: Optional[str] = None
    qa_reviewer_notes: Optional[str] = None
    capa_required: Optional[bool] = None
    capa_action_plan: Optional[str] = None
    assigned_investigator: Optional[str] = None
    status: Optional[str] = None
    confirmed_by_user: Optional[str] = None

class ComplaintResponse(ComplaintBase):
    id: str
    complaint_number: str
    raw_source_type: str
    raw_text: str
    raw_file_name: Optional[str] = None
    ai_extracted_json: Optional[Dict[str, Any]] = None
    ai_risk_level: Optional[str] = None
    ai_risk_rationale: Optional[str] = None
    ai_capa_suggestion: Optional[str] = None
    ai_summary: Optional[str] = None
    ai_completeness_flags: Optional[List[Dict[str, Any]]] = None
    ai_duplicate_warning: Optional[Dict[str, Any]] = None
    ai_model_version: Optional[str] = None
    ai_analyzed_at: Optional[datetime] = None
    confirmed_by_user: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

class ComplaintListResponse(BaseModel):
    total: int
    items: List[ComplaintResponse]
