import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import Base, get_db
from app import crud, schemas

from sqlalchemy.pool import StaticPool

# Setup in-memory test database with StaticPool to retain tables across connections
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "llama-3.1-8b-instant" in data["extraction_model"]

def test_create_and_list_complaint():
    # 1. Create a complaint
    payload = {
        "raw_source_type": "TEXT",
        "raw_text": "Sample test complaint text",
        "customer_name": "Test Hospital",
        "product_name": "Test Product 50mg",
        "product_type": "FDF",
        "dosage_form": "Tablet",
        "batch_number": "TST-9901",
        "complaint_category": "Packaging/Labeling",
        "complaint_description": "Damaged blister packaging on arrival.",
        "severity_classification": "MINOR",
        "capa_required": False
    }
    create_res = client.post("/api/complaints", json=payload)
    assert create_res.status_code == 201
    created_data = create_res.json()
    assert created_data["complaint_number"].startswith("CMP-")
    assert created_data["batch_number"] == "TST-9901"
    
    # 2. List complaints
    list_res = client.get("/api/complaints")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["total"] >= 1
    assert any(c["batch_number"] == "TST-9901" for c in list_data["items"])

def test_duplicate_check():
    # Insert initial complaint
    payload = {
        "raw_source_type": "TEXT",
        "raw_text": "Initial complaint for lot A1",
        "product_name": "Paracetamol 500mg",
        "product_type": "FDF",
        "dosage_form": "Tablet",
        "batch_number": "LOT-A100",
        "complaint_category": "Physical Defect",
        "complaint_description": "Chipped tablet found in blister.",
        "severity_classification": "MINOR"
    }
    client.post("/api/complaints", json=payload)

    # Check duplicate endpoint with exact product + batch
    dup_res = client.get("/api/complaints/check-duplicate?product_name=Paracetamol 500mg&batch_number=LOT-A100")
    assert dup_res.status_code == 200
    dup_data = dup_res.json()
    assert dup_data["is_duplicate"] is True
    assert dup_data["matched_complaint_number"] is not None

    # Check non-duplicate
    non_dup_res = client.get("/api/complaints/check-duplicate?product_name=Paracetamol 500mg&batch_number=LOT-DIFFERENT")
    assert non_dup_res.status_code == 200
    assert non_dup_res.json()["is_duplicate"] is False

def test_analyze_endpoint_critical_complaint():
    # Read sample critical complaint text
    sample_path = os.path.join(os.path.dirname(__file__), "..", "..", "sample_data", "sample_critical_complaint.txt")
    with open(sample_path, "r", encoding="utf-8") as f:
        critical_text = f.read()

    analyze_res = client.post("/api/complaints/analyze", json={
        "raw_text": critical_text,
        "source_type": "EMAIL"
    })
    assert analyze_res.status_code == 200
    data = analyze_res.json()
    
    # Assert structured extraction
    assert data["extracted_fields"]["batch_number"] == "CPX24089"
    assert "Ciprofloxacin" in data["extracted_fields"]["product_name"]
    # Assert risk classification
    assert data["risk_level"] == "CRITICAL"
    # Assert CAPA suggestion and summary are generated
    assert len(data["capa_suggestion"]) > 20
    assert len(data["summary"]) > 20

def test_analyze_endpoint_minor_complaint():
    # Read sample minor complaint text
    sample_path = os.path.join(os.path.dirname(__file__), "..", "..", "sample_data", "sample_minor_complaint.txt")
    with open(sample_path, "r", encoding="utf-8") as f:
        minor_text = f.read()

    analyze_res = client.post("/api/complaints/analyze", json={
        "raw_text": minor_text,
        "source_type": "EMAIL"
    })
    assert analyze_res.status_code == 200
    data = analyze_res.json()
    
    # Assert structured extraction
    assert data["extracted_fields"]["batch_number"] == "MET-API-26-0412"
    assert "Metformin" in data["extracted_fields"]["product_name"]
    # Assert risk classification
    assert data["risk_level"] == "MINOR"
    assert len(data["capa_suggestion"]) > 20
