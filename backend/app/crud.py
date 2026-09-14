from datetime import datetime, timezone
from typing import List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, or_
from app.models import Complaint
from app.schemas import ComplaintCreate, ComplaintUpdate

def get_next_complaint_number(db: Session) -> str:
    """Generates an incremental complaint sequence: CMP-YYYY-XXXX."""
    current_year = datetime.now(timezone.utc).year
    prefix = f"CMP-{current_year}-"
    
    # Query highest number for the current year
    latest = (
        db.query(Complaint)
        .filter(Complaint.complaint_number.like(f"{prefix}%"))
        .order_by(desc(Complaint.complaint_number))
        .first()
    )
    
    if latest:
        try:
            seq = int(latest.complaint_number.split("-")[-1]) + 1
        except (ValueError, IndexError):
            seq = 1
    else:
        seq = 1
        
    return f"{prefix}{seq:04d}"

def create_complaint(db: Session, complaint_in: ComplaintCreate) -> Complaint:
    complaint_data = complaint_in.model_dump()
    complaint_data["complaint_number"] = get_next_complaint_number(db)
    
    db_obj = Complaint(**complaint_data)
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj

def get_complaint(db: Session, complaint_id: str) -> Optional[Complaint]:
    return db.query(Complaint).filter(Complaint.id == complaint_id).first()

def get_complaints(
    db: Session,
    skip: int = 0,
    limit: int = 50,
    search: Optional[str] = None,
    risk: Optional[str] = None,
    status: Optional[str] = None,
) -> Tuple[int, List[Complaint]]:
    query = db.query(Complaint)
    
    if search:
        search_pattern = f"%{search}%"
        query = query.filter(
            or_(
                Complaint.complaint_number.ilike(search_pattern),
                Complaint.product_name.ilike(search_pattern),
                Complaint.batch_number.ilike(search_pattern),
                Complaint.customer_name.ilike(search_pattern),
                Complaint.complaint_description.ilike(search_pattern),
            )
        )
    
    if risk:
        query = query.filter(Complaint.severity_classification == risk.upper())
        
    if status:
        query = query.filter(Complaint.status == status.upper())
        
    total = query.count()
    items = query.order_by(desc(Complaint.created_at)).offset(skip).limit(limit).all()
    return total, items

def update_complaint(db: Session, complaint_id: str, update_in: ComplaintUpdate) -> Optional[Complaint]:
    db_obj = get_complaint(db, complaint_id)
    if not db_obj:
        return None
        
    update_data = update_in.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_obj, key, value)
        
    db_obj.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(db_obj)
    return db_obj

def delete_complaint(db: Session, complaint_id: str) -> bool:
    db_obj = get_complaint(db, complaint_id)
    if not db_obj:
        return False
    db.delete(db_obj)
    db.commit()
    return True

def find_duplicate_complaint(
    db: Session,
    product_name: Optional[str],
    batch_number: Optional[str],
    exclude_id: Optional[str] = None
) -> Optional[Complaint]:
    """
    Checks if a complaint already exists with matching product and batch number.
    In pharma GxP, duplicate complaints for the same batch indicate either repeat
    reporting or widespread batch defect clusters.
    """
    if not product_name or not batch_number:
        return None
        
    p_norm = product_name.strip().lower()
    b_norm = batch_number.strip().lower()
    
    query = db.query(Complaint).filter(
        func.lower(Complaint.product_name) == p_norm,
        func.lower(Complaint.batch_number) == b_norm
    )
    
    if exclude_id:
        query = query.filter(Complaint.id != exclude_id)
        
    return query.order_by(desc(Complaint.created_at)).first()
