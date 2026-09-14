import json
from typing import TypedDict, List, Dict, Any, Optional
from sqlalchemy.orm import Session
from langgraph.graph import StateGraph, START, END

from app.config import settings
from app.services.groq_client import call_groq_json
from app.crud import find_duplicate_complaint

# =====================================================================
# 1. STATE DEFINITION
# =====================================================================

class ComplaintState(TypedDict):
    raw_text: str
    source_type: str
    extracted_fields: Dict[str, Any]
    completeness_flags: List[Dict[str, Any]]
    duplicate_warning: Dict[str, Any]
    risk_level: str
    risk_rationale: str
    capa_suggestion: str
    summary: str
    # Internal context passed to nodes
    db_session: Optional[Session]


# =====================================================================
# 2. NODE IMPLEMENTATIONS
# =====================================================================

def node_extract_fields(state: ComplaintState) -> Dict[str, Any]:
    """
    NODE 1: Structured Entity Extraction
    MODEL: llama-3.1-8b-instant
    WHY: Low-latency (~200ms) and highly cost-effective for entity recognition.
    Extracts customer name, product, batch number, dosage form, dates, and category.
    """
    system_prompt = """You are a pharmaceutical Quality Assurance (QA) data extraction assistant.
Extract structured entities from the raw customer complaint text into valid JSON with these exact keys:
- customer_name: string or null
- customer_contact: string (email/phone) or null
- product_name: string or null
- product_type: "API" (Active Pharmaceutical Ingredient) or "FDF" (Finished Dosage Form) or null
- dosage_form: string (e.g., 'Sterile Injectable Vial', 'Tablet', 'Oral Solution', 'API Powder') or null
- batch_number: string or null
- manufacture_date: string (YYYY-MM-DD) or null
- expiry_date: string (YYYY-MM-DD) or null
- complaint_date: string (YYYY-MM-DD) or null
- complaint_category: one of ["Contamination/Foreign Matter", "Packaging/Labeling", "Physical Defect", "Adverse Event/Potency", "API Quality/Assay", "Other"]
- complaint_description: concise summary of the reported defect or problem

Output ONLY a single valid JSON object."""

    user_prompt = f"Raw Complaint Ingestion:\n{state['raw_text']}"
    
    extracted = call_groq_json(
        model=settings.GROQ_EXTRACTION_MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )
    
    return {"extracted_fields": extracted}


def node_completeness_check(state: ComplaintState) -> Dict[str, Any]:
    """
    NODE 2: Completeness & Ambiguity Evaluation
    MODEL: llama-3.1-8b-instant
    WHY: High-speed rule-checking against GMP compliance expectations.
    Flags missing mandatory fields (product, batch, dates, detailed description).
    """
    system_prompt = """You are a GMP Compliance Auditor. Review the extracted fields against the raw complaint.
Determine if any mandatory fields are missing or ambiguous for a formal pharma QA investigation.
Mandatory fields for pharma complaint logging:
1. product_name (Critical)
2. batch_number (Critical for lot traceability & quarantine)
3. complaint_description (Must specify observable defect)
4. expiry_date / manufacture_date (Important for stability assessment)
5. customer_name / contact (Important for complaint follow-up)

Return a JSON object with:
- flags: an array of objects, each having:
  - field: name of the field (e.g. 'batch_number')
  - status: 'missing' or 'ambiguous'
  - note: explanation of why it is flagged and what QA needs to request from the customer

If all critical information is present, return {"flags": []}."""

    user_prompt = f"Raw Complaint:\n{state['raw_text']}\n\nExtracted Fields:\n{json.dumps(state['extracted_fields'], indent=2)}"

    result = call_groq_json(
        model=settings.GROQ_EXTRACTION_MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )
    
    flags = result.get("flags", [])
    return {"completeness_flags": flags}


def node_duplicate_check(state: ComplaintState) -> Dict[str, Any]:
    """
    NODE 3: Duplicate & Batch Cluster Detector
    ENGINE: Deterministic Database Query (Python/SQL)
    WHY: Zero LLM token waste. Uses SQL exact/normalized match on product_name + batch_number.
    In pharma, a matching product + batch indicates either a duplicated intake or a serious
    cluster of complaints on a single manufactured lot that may warrant a recall.
    """
    db = state.get("db_session")
    product_name = state["extracted_fields"].get("product_name")
    batch_number = state["extracted_fields"].get("batch_number")
    
    duplicate_info = {
        "is_duplicate": False,
        "matched_complaint_id": None,
        "matched_complaint_number": None,
        "confidence": 0.0,
        "details": "No existing complaints found for this Product and Batch combination."
    }
    
    if db and product_name and batch_number:
        matched = find_duplicate_complaint(db, product_name, batch_number)
        if matched:
            duplicate_info = {
                "is_duplicate": True,
                "matched_complaint_id": matched.id,
                "matched_complaint_number": matched.complaint_number,
                "confidence": 0.95,
                "details": f"Existing complaint {matched.complaint_number} logged on {matched.created_at.strftime('%Y-%m-%d')} shares the same Product ('{matched.product_name}') and Batch ('{matched.batch_number}'). Review for lot-level trend analysis or duplicate entry."
            }
            
    return {"duplicate_warning": duplicate_info}


def node_risk_classification(state: ComplaintState) -> Dict[str, Any]:
    """
    NODE 4: Regulatory Risk Classification & Rationale
    MODEL: llama-3.3-70b-versatile
    WHY: Risk classification under ICH Q9 Quality Risk Management and FDA 21 CFR Part 211
    requires sophisticated pharmaceutical judgment. Distinguishing between life-threatening
    sterility/potency defects (Critical) vs packaging/labeling cosmetic issues (Minor) requires
    deep domain reasoning that smaller 8B models can hallucinate or underestimate.
    """
    system_prompt = """You are a Senior Pharmaceutical Quality Assurance Director and Risk Assessor.
Analyze the complaint and classify its risk severity according to ICH Q9 / FDA / WHO GMP guidelines into one of three levels:

1. CRITICAL:
   - Defects that may be life-threatening or could cause serious health hazards (e.g., contamination, visible particulates in injectables, sterility breach, mix-up of chemical active ingredients, super-potency / sub-potency causing adverse events).
2. MAJOR:
   - Defects that could impair product efficacy or cause illness, but are not immediately life-threatening (e.g., missing leaflet with critical dosing info, physical tablet breakage exceeding friability specs, non-critical assay variations, dissolution failures).
3. MINOR:
   - Defects that have negligible impact on patient safety or product efficacy (e.g., secondary outer carton abrasion, smudged outer shipping label barcode, cosmetic bottle dent with primary seal intact).

Return a JSON object with:
- risk_level: exactly "CRITICAL", "MAJOR", or "MINOR"
- risk_rationale: a concise, defensible 2-3 sentence regulatory justification explaining the clinical and compliance basis for this rating."""

    user_prompt = f"""Complaint Details:
Product: {state['extracted_fields'].get('product_name')} ({state['extracted_fields'].get('product_type', 'Unknown')})
Dosage Form: {state['extracted_fields'].get('dosage_form')}
Batch: {state['extracted_fields'].get('batch_number')}
Category: {state['extracted_fields'].get('complaint_category')}
Description: {state['extracted_fields'].get('complaint_description')}

Raw Intake:
{state['raw_text']}"""

    result = call_groq_json(
        model=settings.GROQ_REASONING_MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )
    
    risk_level = result.get("risk_level", "MAJOR").upper()
    if risk_level not in ["CRITICAL", "MAJOR", "MINOR"]:
        risk_level = "MAJOR"
        
    return {
        "risk_level": risk_level,
        "risk_rationale": result.get("risk_rationale", "Standard regulatory review required.")
    }


def node_capa_suggestion(state: ComplaintState) -> Dict[str, Any]:
    """
    NODE 5: CAPA (Corrective and Preventive Action) Direction
    MODEL: llama-3.3-70b-versatile
    WHY: CAPA formulation requires manufacturing process insight (e.g., sterile filtration,
    environmental monitoring, blister sealing parameters, supplier QA). The 70B model
    synthesizes realistic, audit-ready investigation steps.
    """
    system_prompt = """You are a Pharma Quality Systems & CAPA Specialist.
Based on the complaint defect and the determined risk level, suggest an immediate CAPA (Corrective and Preventive Action) roadmap for the QA investigation team.

Structure the recommendation into:
1. Immediate Containment (e.g. batch quarantine, retain sample inspection, distribution hold)
2. Root Cause Investigation (e.g. line logs, environmental monitoring, filtration integrity, operator training)
3. Preventive Action / Systemic Fix (e.g. equipment recalibration, packaging material spec revision, supplier audit)
4. Regulatory Notification consideration (e.g., FDA Field Alert Report within 3 working days if Critical)

Return a JSON object with:
- capa_suggestion: a numbered, concise, actionable list formatted as clean text."""

    user_prompt = f"""Risk Level: {state['risk_level']}
Risk Rationale: {state['risk_rationale']}
Product: {state['extracted_fields'].get('product_name')} ({state['extracted_fields'].get('dosage_form')})
Batch: {state['extracted_fields'].get('batch_number')}
Category: {state['extracted_fields'].get('complaint_category')}
Defect: {state['extracted_fields'].get('complaint_description')}"""

    result = call_groq_json(
        model=settings.GROQ_REASONING_MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )
    
    return {"capa_suggestion": result.get("capa_suggestion", "Initiate standard retain sample testing and batch record review.")}


def node_executive_summary(state: ComplaintState) -> Dict[str, Any]:
    """
    NODE 6: Executive Triage Summary
    MODEL: llama-3.1-8b-instant
    WHY: Text condensation of already structured findings into a single executive paragraph.
    Fast and concise.
    """
    system_prompt = """You are an Executive QA Medical Writer.
Write a clear, professional, one-paragraph executive summary of this customer complaint for senior management and the Quality Review Board.
The paragraph should concisely state:
- Who reported the complaint and when
- The affected product, dosage form, and batch
- The nature of the reported defect
- The assessed risk level and immediate containment priority

Return a JSON object with:
- summary: a single polished paragraph (under 120 words)."""

    user_prompt = f"""Customer: {state['extracted_fields'].get('customer_name')}
Product: {state['extracted_fields'].get('product_name')}
Batch: {state['extracted_fields'].get('batch_number')}
Category: {state['extracted_fields'].get('complaint_category')}
Risk Level: {state['risk_level']}
Rationale: {state['risk_rationale']}
Description: {state['extracted_fields'].get('complaint_description')}"""

    result = call_groq_json(
        model=settings.GROQ_EXTRACTION_MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )
    
    return {"summary": result.get("summary", "Customer complaint processed and queued for QA review.")}


# =====================================================================
# 3. GRAPH COMPILATION
# =====================================================================

def build_complaint_pipeline():
    """Builds and compiles the 6-node LangGraph pipeline."""
    graph = StateGraph(ComplaintState)
    
    # Add nodes (using distinct names from state keys)
    graph.add_node("extract_fields", node_extract_fields)
    graph.add_node("check_completeness", node_completeness_check)
    graph.add_node("check_duplicates", node_duplicate_check)
    graph.add_node("classify_risk", node_risk_classification)
    graph.add_node("suggest_capa", node_capa_suggestion)
    graph.add_node("generate_summary", node_executive_summary)
    
    # Define linear sequential flow
    graph.add_edge(START, "extract_fields")
    graph.add_edge("extract_fields", "check_completeness")
    graph.add_edge("check_completeness", "check_duplicates")
    graph.add_edge("check_duplicates", "classify_risk")
    graph.add_edge("classify_risk", "suggest_capa")
    graph.add_edge("suggest_capa", "generate_summary")
    graph.add_edge("generate_summary", END)
    
    return graph.compile()

# Global compiled pipeline instance
complaint_pipeline = build_complaint_pipeline()

def run_complaint_analysis(
    raw_text: str,
    source_type: str = "TEXT",
    db_session: Optional[Session] = None
) -> Dict[str, Any]:
    """
    Executes the LangGraph state pipeline synchronously and returns the final state.
    """
    initial_state: ComplaintState = {
        "raw_text": raw_text,
        "source_type": source_type,
        "extracted_fields": {},
        "completeness_flags": [],
        "duplicate_warning": {},
        "risk_level": "MINOR",
        "risk_rationale": "",
        "capa_suggestion": "",
        "summary": "",
        "db_session": db_session
    }
    
    final_state = complaint_pipeline.invoke(initial_state)
    return final_state
