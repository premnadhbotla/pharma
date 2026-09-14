from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form, status
from sqlalchemy.orm import Session

from app.database import get_db
from app import crud, schemas
from app.services.document_parser import extract_text_from_file
from app.services.pipeline import run_complaint_analysis

router = APIRouter(prefix="/api/complaints", tags=["Complaints"])

# --- 1. COPILOT ANALYSIS ENDPOINTS ---

@router.post("/analyze", response_model=schemas.AIAnalysisResult)
async def analyze_complaint_text(
    payload: schemas.AnalyzeTextRequest,
    db: Session = Depends(get_db)
):
    """
    Executes the 6-stage LangGraph copilot pipeline on raw complaint text.
    Returns extracted fields, completeness flags, duplicate warning, risk classification,
    CAPA suggestions, and executive summary.
    """
    if not payload.raw_text or not payload.raw_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Complaint text cannot be empty."
        )
        
    analysis_state = run_complaint_analysis(
        raw_text=payload.raw_text.strip(),
        source_type=payload.source_type,
        db_session=db
    )
    
    return schemas.AIAnalysisResult(
        extracted_fields=schemas.ExtractedEntities(**analysis_state.get("extracted_fields", {})),
        completeness_flags=[schemas.CompletenessFlag(**f) for f in analysis_state.get("completeness_flags", [])],
        duplicate_warning=schemas.DuplicateWarning(**analysis_state.get("duplicate_warning", {})),
        risk_level=analysis_state.get("risk_level", "MAJOR"),
        risk_rationale=analysis_state.get("risk_rationale", ""),
        capa_suggestion=analysis_state.get("capa_suggestion", ""),
        summary=analysis_state.get("summary", "")
    )


@router.post("/analyze-file", response_model=schemas.AIAnalysisResult)
async def analyze_complaint_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Extracts text from an uploaded PDF, TXT, or EML document, then triggers the LangGraph analysis.
    """
    content_bytes = await file.read()
    if not content_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )
        
    extracted_text = extract_text_from_file(content_bytes, file.filename)
    if not extracted_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not extract readable text from the provided file."
        )
        
    source_type = "FILE_PDF" if file.filename.lower().endswith(".pdf") else "EMAIL"
    
    analysis_state = run_complaint_analysis(
        raw_text=extracted_text,
        source_type=source_type,
        db_session=db
    )
    
    return schemas.AIAnalysisResult(
        extracted_fields=schemas.ExtractedEntities(**analysis_state.get("extracted_fields", {})),
        completeness_flags=[schemas.CompletenessFlag(**f) for f in analysis_state.get("completeness_flags", [])],
        duplicate_warning=schemas.DuplicateWarning(**analysis_state.get("duplicate_warning", {})),
        risk_level=analysis_state.get("risk_level", "MAJOR"),
        risk_rationale=analysis_state.get("risk_rationale", ""),
        capa_suggestion=analysis_state.get("capa_suggestion", ""),
        summary=analysis_state.get("summary", "")
    )


@router.get("/check-duplicate", response_model=schemas.DuplicateWarning)
def check_duplicate_complaint(
    product_name: str = Query(..., description="Product name to check"),
    batch_number: str = Query(..., description="Batch number to check"),
    exclude_id: Optional[str] = Query(None, description="Complaint ID to exclude from match"),
    db: Session = Depends(get_db)
):
    """
    Lightweight endpoint for frontend real-time duplicate validation during manual form edits.
    """
    match = crud.find_duplicate_complaint(db, product_name, batch_number, exclude_id)
    if match:
        return schemas.DuplicateWarning(
            is_duplicate=True,
            matched_complaint_id=match.id,
            matched_complaint_number=match.complaint_number,
            confidence=0.95,
            details=f"Lot {match.batch_number} for product {match.product_name} was previously logged in complaint {match.complaint_number}."
        )
    return schemas.DuplicateWarning(
        is_duplicate=False,
        confidence=0.0,
        details="No duplicate record detected."
    )


# --- 2. COMPLAINTS CRUD ENDPOINTS ---

@router.get("", response_model=schemas.ComplaintListResponse)
def list_complaints(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    search: Optional[str] = None,
    risk: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Lists all complaints with optional filtering and search."""
    total, items = crud.get_complaints(
        db=db,
        skip=skip,
        limit=limit,
        search=search,
        risk=risk,
        status=status
    )
    return schemas.ComplaintListResponse(total=total, items=items)


@router.post("", response_model=schemas.ComplaintResponse, status_code=status.HTTP_201_CREATED)
def create_complaint(
    complaint_in: schemas.ComplaintCreate,
    db: Session = Depends(get_db)
):
    """Creates a new human-confirmed complaint record in the GxP database."""
    created = crud.create_complaint(db, complaint_in)
    return created


@router.get("/{complaint_id}", response_model=schemas.ComplaintResponse)
def get_complaint(
    complaint_id: str,
    db: Session = Depends(get_db)
):
    """Retrieves a single complaint by its unique ID."""
    complaint = crud.get_complaint(db, complaint_id)
    if not complaint:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Complaint with ID '{complaint_id}' not found."
        )
    return complaint


@router.put("/{complaint_id}", response_model=schemas.ComplaintResponse)
def update_complaint(
    complaint_id: str,
    update_in: schemas.ComplaintUpdate,
    db: Session = Depends(get_db)
):
    """Updates an existing complaint record."""
    updated = crud.update_complaint(db, complaint_id, update_in)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Complaint with ID '{complaint_id}' not found."
        )
    return updated


@router.delete("/{complaint_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_complaint(
    complaint_id: str,
    db: Session = Depends(get_db)
):
    """Deletes a complaint record."""
    success = crud.delete_complaint(db, complaint_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Complaint with ID '{complaint_id}' not found."
        )
    return None
