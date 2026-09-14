from datetime import date, datetime, timezone
from sqlalchemy.orm import Session
from app.models import Complaint
from app.crud import get_next_complaint_number

def seed_baseline_complaints(db: Session):
    """
    Seeds initial realistic pharma complaints into the database if empty.
    This enables immediate demonstration of duplicate/batch-cluster warnings!
    """
    count = db.query(Complaint).count()
    if count > 0:
        return

    # Seed 1: A prior logged complaint for Metformin API Batch MET-API-26-0412
    c1 = Complaint(
        complaint_number=get_next_complaint_number(db),
        status="UNDER_REVIEW",
        created_at=datetime.now(timezone.utc),
        raw_source_type="EMAIL",
        raw_text="Preliminary delivery notice from Novis Formulation Labs reporting outer barcode issue on Metformin API drum.",
        product_name="Metformin Hydrochloride USP API",
        product_type="API",
        dosage_form="API Crystalline Powder",
        batch_number="MET-API-26-0412",
        manufacture_date=date(2026, 6, 1),
        expiry_date=date(2029, 6, 1),
        complaint_date=date(2026, 9, 8),
        complaint_category="Packaging/Labeling",
        complaint_description="Warehouse barcode scan failure on drum #12 from transit abrasion.",
        severity_classification="MINOR",
        qa_reviewer_notes="Customer reported receiving 25 drums, barcode unreadable on pallet A.",
        capa_required=False,
        confirmed_by_user="QA Specialist Sandra Lin"
    )
    db.add(c1)
    db.commit()

    # Seed 2: A resolved complaint for another product
    c2 = Complaint(
        complaint_number=get_next_complaint_number(db),
        status="CLOSED",
        created_at=datetime.now(timezone.utc),
        raw_source_type="TEXT",
        raw_text="Hospital pharmacy noted slight discoloration on carton exterior of Atorvastatin 20mg.",
        product_name="Atorvastatin 20mg Film-Coated Tablets",
        product_type="FDF",
        dosage_form="Film-Coated Tablet",
        batch_number="ATV-26-1088",
        manufacture_date=date(2026, 1, 10),
        expiry_date=date(2028, 1, 10),
        complaint_date=date(2026, 8, 14),
        complaint_category="Packaging/Labeling",
        complaint_description="Carton exterior water droplet staining during monsoon freight transit. Blister foils unaffected.",
        severity_classification="MINOR",
        qa_reviewer_notes="Blister seal integrity verified 100%. No ingress. Carrier carton wrapped with moisture barrier.",
        capa_required=True,
        capa_action_plan="Replaced outer corrugated box moisture shrink-wrap supplier.",
        confirmed_by_user="QA Manager David Chen"
    )
    db.add(c2)
    db.commit()
    print("Successfully seeded baseline pharma complaints.")
