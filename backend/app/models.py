import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, DateTime, Date, Boolean, JSON
from app.database import Base

def generate_uuid():
    return str(uuid.uuid4())

def utc_now():
    return datetime.now(timezone.utc)

class Complaint(Base):
    __tablename__ = "complaints"

    # --- IDENTIFICATION & WORKFLOW STATUS ---
    id = Column(String(36), primary_key=True, default=generate_uuid)
    complaint_number = Column(String(32), unique=True, nullable=False, index=True)
    status = Column(String(32), nullable=False, default="DRAFT")  # DRAFT, UNDER_REVIEW, LOGGED, CLOSED
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    # --- 1. RAW INGESTION DATA (Immutable Intake) ---
    raw_source_type = Column(String(16), nullable=False, default="TEXT")  # 'TEXT', 'EMAIL', 'FILE_PDF'
    raw_text = Column(Text, nullable=False)
    raw_file_name = Column(String(255), nullable=True)

    # --- 2. AI-EXTRACTED / COPILOT METADATA (Advisory Snapshot) ---
    ai_extracted_json = Column(JSON, nullable=True)
    ai_risk_level = Column(String(16), nullable=True)  # 'CRITICAL', 'MAJOR', 'MINOR'
    ai_risk_rationale = Column(Text, nullable=True)
    ai_capa_suggestion = Column(Text, nullable=True)
    ai_summary = Column(Text, nullable=True)
    ai_completeness_flags = Column(JSON, nullable=True)  # List of {field, status, note}
    ai_duplicate_warning = Column(JSON, nullable=True)   # {is_duplicate, matched_id, confidence, details}
    ai_model_version = Column(String(64), nullable=True)
    ai_analyzed_at = Column(DateTime(timezone=True), nullable=True)

    # --- 3. HUMAN-CONFIRMED QMS FIELDS (Official GxP Record of Truth) ---
    customer_name = Column(String(255), nullable=True)
    customer_contact = Column(String(255), nullable=True)
    product_name = Column(String(255), nullable=False, index=True)
    product_type = Column(String(32), nullable=True)  # 'API', 'FDF'
    dosage_form = Column(String(128), nullable=True)  # e.g. 'Tablet', 'Sterile Injectable', 'API Powder'
    batch_number = Column(String(128), nullable=False, index=True)
    manufacture_date = Column(Date, nullable=True)
    expiry_date = Column(Date, nullable=True)
    complaint_date = Column(Date, nullable=True)
    complaint_category = Column(String(64), nullable=False)  # 'Packaging/Labeling', 'Contamination/Foreign Matter', etc.
    complaint_description = Column(Text, nullable=False)
    severity_classification = Column(String(16), nullable=False)  # 'CRITICAL', 'MAJOR', 'MINOR'
    qa_reviewer_notes = Column(Text, nullable=True)
    capa_required = Column(Boolean, default=False)
    capa_action_plan = Column(Text, nullable=True)
    assigned_investigator = Column(String(255), nullable=True)
    confirmed_by_user = Column(String(128), nullable=True)
