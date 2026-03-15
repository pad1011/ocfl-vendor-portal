"""
OCFL Procurement Purchase Request Portal — FastAPI Backend

Provides endpoints for:
  - Uploading vendor quotes (PDF/image), extracting fields via local LLM (Ollama)
  - Running compliance checks against Orange County procurement rules
  - Retrieving and listing submissions
  - Generating filled PDF procurement forms
"""

from __future__ import annotations

import base64
import json
import uuid
from datetime import datetime, date
from pathlib import Path
from typing import Any, Optional

import pdfplumber
import uvicorn
import httpx
import os
import logging

logger = logging.getLogger(__name__)
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="OCFL Procurement Portal API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
FILLED_DIR = BASE_DIR / "filled"
DB_PATH = BASE_DIR / "db.json"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

UPLOADS_DIR.mkdir(exist_ok=True)
FILLED_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# LLM Configuration — uses Ollama (local) by default
# Set OLLAMA_HOST to point to your Ollama server (default: http://localhost:11434)
# Set LLM_MODEL to choose the model (default: llama3.1:70b)
# ---------------------------------------------------------------------------
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.1:70b")
SAM_API_KEY = os.environ.get("SAM_API_KEY", "")

# ---------------------------------------------------------------------------
# JSON file "database"
# ---------------------------------------------------------------------------


def _load_db() -> dict:
    if DB_PATH.exists():
        return json.loads(DB_PATH.read_text())
    return {"submissions": {}}


def _save_db(db: dict) -> None:
    DB_PATH.write_text(json.dumps(db, indent=2, default=str))


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class LineItem(BaseModel):
    description: str = ""
    quantity: float | int = 0
    unit_price: float = 0.0
    extended_price: float = 0.0


class ExtractedData(BaseModel):
    vendor_name: str = ""
    vendor_number: str = ""
    contract_number: str = ""
    requisition_number: str = ""
    date_submitted: str = ""
    requestor_name: str = ""
    requestor_phone: str = ""
    department: str = ""
    line_items: list[LineItem] = Field(default_factory=list)
    total_amount: float = 0.0
    contract_type: str = "standard"
    scope_of_services: str = ""
    insurance_mentioned: bool = False
    expiration_date: str = ""
    funding_type: str = "standard"
    # --- Fields from MS Forms Purchase Request (user-entered) ---
    accounting_line: str = ""
    type_of_request: str = ""  # Hardware, Software/Licensing, Contract Labor, Other
    future_cost_hardware: str = ""  # yes / no
    future_cost_software: str = ""  # yes / no
    renewal_cost_year1: str = ""  # estimated year one renewal cost
    custodian_code_shipping_notes: str = ""  # custodian code, shipping location, other notes


class ComplianceIssue(BaseModel):
    rule_id: str
    severity: str  # critical / warning / info
    message: str
    section_reference: str
    suggested_action: str


class ComplianceResult(BaseModel):
    issues: list[ComplianceIssue] = Field(default_factory=list)
    threshold_category: str = ""
    board_approval_required: bool = False
    procurement_method: str = ""


class SubmissionResponse(BaseModel):
    submission_id: str
    filename: str
    uploaded_at: str
    extracted_data: ExtractedData
    compliance: ComplianceResult


class ComplianceCheckRequest(BaseModel):
    submission_id: str
    extracted_data: ExtractedData


class GenerateFormRequest(BaseModel):
    submission_id: str
    form_type: str = "piggyback_checklist"
    field_overrides: dict[str, Any] = Field(default_factory=dict)


class ContractSearchRequest(BaseModel):
    query: str
    search_type: str = "vendor"  # "vendor" or "product"


class AuditEvent(BaseModel):
    """A single event in a submission's timeline."""
    event_type: str  # created, status_change, compliance_check, details_updated, routing, note_added, approval, rejection
    timestamp: str
    actor: str = "system"  # system, user, or specific name
    description: str
    metadata: dict = Field(default_factory=dict)


class ApprovalRoute(BaseModel):
    """Routing destination based on thresholds."""
    route_to: str  # e.g., "supervisor", "procurement", "bcc"
    reason: str
    threshold_category: str
    required_approvals: list[str] = Field(default_factory=list)
    estimated_lead_time: str = ""


class DuplicateAlert(BaseModel):
    """Alert for potential duplicate or related purchases."""
    alert_type: str  # duplicate, related, consolidation_opportunity
    severity: str  # info, warning, critical
    message: str
    related_submissions: list[dict] = Field(default_factory=list)
    combined_total: float = 0.0
    suggested_action: str = ""


class NLQueryRequest(BaseModel):
    """Natural language query request."""
    question: str


# ---------------------------------------------------------------------------
# AI extraction
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You are an expert at reading vendor quotes and procurement documents for \
Orange County, Florida (OCFL). Analyze the provided document and extract \
the following fields. Return ONLY valid JSON with no markdown fencing.

Fields to extract:
- vendor_name: The vendor / supplier company name
- vendor_number: Vendor ID number if present (else "")
- contract_number: Contract or agreement number if referenced
- requisition_number: Requisition number if present
- date_submitted: Date on the quote or document (ISO format YYYY-MM-DD if possible)
- requestor_name: The county employee or requestor name if shown
- requestor_phone: Requestor phone number if shown
- department: Orange County department or division name
- line_items: Array of objects with {description, quantity, unit_price, extended_price}
  - quantity and prices should be numbers; extended_price = quantity * unit_price
- total_amount: Total dollar amount of the quote/purchase (number)
- contract_type: One of "piggyback", "state_contract", "gsa", "cooperative", \
"standard", "emergency", "sole_source". Infer from context clues such as \
references to other agency contracts, GSA schedules, emergency language, etc.
- scope_of_services: Brief description of what is being purchased or the \
services to be performed
- insurance_mentioned: true if the document mentions insurance, COI, \
certificates of insurance, or indemnification; false otherwise
- expiration_date: Contract or quote expiration date if mentioned (ISO format)
- funding_type: One of "standard", "federal", "state". Look for mentions of \
federal grants, FEMA, federal funding, state grants, etc. Default "standard".

Context: Orange County FL uses the Advantage financial system. Procurement \
types include piggyback (using another agency's contract), state contracts, \
GSA schedules, cooperative purchasing, standard competitive quotes, emergency \
purchases, and sole source. The Piggyback Requisition Checklist (Exhibit 32) \
and Expedited Quoting Form (Exhibit 40) are commonly referenced forms.

Return the JSON object exactly matching the field names above.\
"""


def _extract_text_from_pdf(file_path: Path) -> str:
    """Extract text content from a PDF using pdfplumber."""
    text_parts: list[str] = []
    with pdfplumber.open(str(file_path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n\n".join(text_parts)


def _call_llm_text(text: str) -> dict:
    """Send extracted text to the local LLM for field extraction."""
    prompt = f"{EXTRACTION_PROMPT}\n\n--- DOCUMENT TEXT ---\n{text}\n--- END ---"
    response = httpx.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "num_predict": 4096},
        },
        timeout=120.0,
    )
    response.raise_for_status()
    return _parse_llm_response(response.json())


def _call_llm_image(file_path: Path, media_type: str) -> dict:
    """Send an image to the local LLM using vision capability."""
    img_b64 = base64.b64encode(file_path.read_bytes()).decode()
    response = httpx.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": EXTRACTION_PROMPT,
                    "images": [img_b64],
                }
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "num_predict": 4096},
        },
        timeout=120.0,
    )
    response.raise_for_status()
    return _parse_llm_response(response.json())


def _parse_llm_response(response_data: dict) -> dict:
    """Parse the LLM response into a dict."""
    raw = response_data["message"]["content"].strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = lines[1:]  # remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Compliance engine
# ---------------------------------------------------------------------------


def run_compliance_checks(data: ExtractedData) -> ComplianceResult:
    """Run the full suite of OCFL procurement compliance checks."""
    issues: list[ComplianceIssue] = []
    amount = data.total_amount
    is_federal = data.funding_type == "federal"
    contract_type = data.contract_type

    # ---- Threshold detection ----
    if is_federal:
        if amount <= 10_000:
            category = "micro_purchase"
            method = "P-Card or Single Quotation"
        elif amount <= 250_000:
            category = "simplified_acquisition"
            method = "Request for Quotations (RFQ) – min 3 written quotes"
        else:
            category = "formal_solicitation"
            method = "Formal Sealed Solicitation (IFB/RFP) by Procurement Division"
    else:
        if amount <= 10_000:
            category = "small_purchase"
            method = "P-Card or Single Quotation"
        elif amount <= 150_000:
            category = "informal_quotes"
            method = "Request for Quotations (RFQ) – min 3 written quotes, 1 M/WBE"
        else:
            category = "formal_solicitation"
            method = "Formal Sealed Solicitation (IFB/RFP) by Procurement Division"

    board_approval = amount > 500_000

    # Info: threshold category
    issues.append(
        ComplianceIssue(
            rule_id="THRESHOLD-001",
            severity="info",
            message=(
                f"Total amount ${amount:,.2f} falls under "
                f"{'federal ' if is_federal else ''}{category.replace('_', ' ')} "
                f"threshold. Required method: {method}."
            ),
            section_reference="Exhibit 29 / Exhibit 37 / Exhibit 38",
            suggested_action=f"Use procurement method: {method}",
        )
    )

    if board_approval:
        issues.append(
            ComplianceIssue(
                rule_id="THRESHOLD-002",
                severity="critical",
                message=(
                    f"Amount ${amount:,.2f} exceeds $500,000. "
                    "Board of County Commissioners approval required."
                ),
                section_reference="Section 2, Exhibit 29",
                suggested_action=(
                    "Submit for BCC approval. Allow 4 weeks for Board approval "
                    "(12 weeks total lead time)."
                ),
            )
        )

    if not is_federal and amount > 150_000:
        issues.append(
            ComplianceIssue(
                rule_id="THRESHOLD-003",
                severity="warning",
                message=(
                    f"Amount ${amount:,.2f} exceeds $150,000 mandatory bid limit. "
                    "Must be formally solicited by Procurement Division."
                ),
                section_reference="Section 6, Exhibit 37",
                suggested_action=(
                    "Submit to Procurement Division for formal solicitation."
                ),
            )
        )

    if is_federal and amount > 250_000:
        issues.append(
            ComplianceIssue(
                rule_id="THRESHOLD-004",
                severity="warning",
                message=(
                    f"Federal-funded amount ${amount:,.2f} exceeds $250,000 "
                    "simplified acquisition threshold. Formal solicitation required."
                ),
                section_reference="Exhibit 38, Resolution 2021-M-29",
                suggested_action=(
                    "Submit to Procurement Division for formal solicitation. "
                    "Independent estimate and written cost analysis required."
                ),
            )
        )

    # ---- Federal funding checks ----
    if is_federal:
        issues.append(
            ComplianceIssue(
                rule_id="FED-001",
                severity="warning",
                message=(
                    "Federal funding detected. Additional federal compliance "
                    "requirements apply (SAM.gov check, Exhibit 36, etc.)."
                ),
                section_reference="Section 8, Exhibit 38",
                suggested_action=(
                    "Complete Federal Compliance Documentation Form (Exhibit 36). "
                    "Verify contractor is not excluded in SAM.gov."
                ),
            )
        )

        if contract_type == "piggyback":
            issues.append(
                ComplianceIssue(
                    rule_id="FED-002",
                    severity="critical",
                    message=(
                        "Piggybacking is DISALLOWED for federal-funded procurements. "
                        "Cooperative procurement may be allowed as an alternative."
                    ),
                    section_reference="Section 8, Section 10, Exhibit 38",
                    suggested_action=(
                        "Do not use piggyback. Consider cooperative procurement "
                        "or conduct a new competitive solicitation."
                    ),
                )
            )

        if amount > 2_000:
            issues.append(
                ComplianceIssue(
                    rule_id="FED-003",
                    severity="info",
                    message=(
                        "If this is a construction procurement, Davis-Bacon Act "
                        "compliance is required for federal-funded construction "
                        "over $2,000."
                    ),
                    section_reference="Section 8, Exhibit 38",
                    suggested_action=(
                        "If construction: include prevailing wage determination "
                        "in solicitation and validate 10 days prior to closing."
                    ),
                )
            )

        if amount > 10_000:
            issues.append(
                ComplianceIssue(
                    rule_id="FED-004",
                    severity="warning",
                    message=(
                        "Federal procurement over $10,000 requires profit "
                        "negotiation as separate element if no price competition."
                    ),
                    section_reference="Section 8, Exhibit 38",
                    suggested_action=(
                        "If sole source or single response: document profit "
                        "negotiation per Exhibit 3."
                    ),
                )
            )

    # ---- Piggyback / ACS checks ----
    if contract_type == "piggyback":
        if not data.contract_number:
            issues.append(
                ComplianceIssue(
                    rule_id="PB-001",
                    severity="critical",
                    message=(
                        "Piggyback purchase requires a contract number. "
                        "No contract number was found in the document."
                    ),
                    section_reference="Section 10, Exhibit 32",
                    suggested_action=(
                        "Provide the contract number for the Alternate Contract "
                        "Source being piggybacked."
                    ),
                )
            )

        if data.expiration_date:
            try:
                exp = datetime.strptime(data.expiration_date, "%Y-%m-%d").date()
                if exp < date.today():
                    issues.append(
                        ComplianceIssue(
                            rule_id="PB-002",
                            severity="critical",
                            message=(
                                f"Contract expiration date {data.expiration_date} "
                                "has passed. Piggyback contract must be active."
                            ),
                            section_reference="Section 10, Exhibit 32 Step 1",
                            suggested_action=(
                                "Verify contract is still active. If expired, "
                                "a new solicitation or different contract is needed."
                            ),
                        )
                    )
            except ValueError:
                pass
        else:
            issues.append(
                ComplianceIssue(
                    rule_id="PB-003",
                    severity="warning",
                    message=(
                        "No contract expiration date found. Verify the piggyback "
                        "contract is not expired before proceeding."
                    ),
                    section_reference="Section 10, Exhibit 32 Step 1",
                    suggested_action=(
                        "Confirm contract is on the ACS log and is not expired."
                    ),
                )
            )

        issues.append(
            ComplianceIssue(
                rule_id="PB-004",
                severity="warning",
                message=(
                    "Piggyback requisition requires: contract on ACS log, "
                    "vendor name match between contract and quote, and "
                    "line-by-line pricing validation."
                ),
                section_reference="Section 10, Exhibit 32 Steps 1-3",
                suggested_action=(
                    "Complete Piggyback Requisition Checklist (Exhibit 32). "
                    "Attach highlighted price sheets matching contractual prices."
                ),
            )
        )

    if contract_type == "gsa" and amount > 50_000:
        issues.append(
            ComplianceIssue(
                rule_id="GSA-001",
                severity="warning",
                message=(
                    "GSA Schedule procurement over $50,000 requires a written "
                    "price analysis memorandum demonstrating pricing advantages."
                ),
                section_reference="Section 10, Exhibit 32 Step 4",
                suggested_action=(
                    "Prepare price analysis memo in writing. Obtain signed "
                    "statement from contract holder on letterhead authorizing "
                    "County use (unless cooperative procurement)."
                ),
            )
        )

    # ---- Insurance check ----
    if not data.insurance_mentioned and data.scope_of_services:
        scope_lower = data.scope_of_services.lower()
        service_keywords = [
            "service", "install", "construct", "repair", "maintain",
            "deliver", "on-site", "county facility", "county property",
        ]
        if any(kw in scope_lower for kw in service_keywords):
            issues.append(
                ComplianceIssue(
                    rule_id="INS-001",
                    severity="warning",
                    message=(
                        "Services may be performed on County property but no "
                        "insurance or COI was mentioned in the document. "
                        "Insurance is required for construction, services at "
                        "County facilities, and services where liability is an issue."
                    ),
                    section_reference="Section 10, Exhibit 33, Exhibit 37/38",
                    suggested_action=(
                        "Verify insurance requirements. Obtain Certificate of "
                        "Insurance with Orange County BCC as Additional Insured."
                    ),
                )
            )

    # ---- Required documents check ----
    if contract_type == "sole_source":
        issues.append(
            ComplianceIssue(
                rule_id="DOC-001",
                severity="critical",
                message=(
                    "Sole Source procurement requires Sole Source Procurement "
                    "Data Sheet (Exhibit 2). Price Negotiation Memorandum "
                    "(Exhibit 3) may also be required."
                ),
                section_reference="Section 4, Exhibit 2, Exhibit 3",
                suggested_action=(
                    "Complete and attach Exhibit 2. If over $500K or federal "
                    "funded, also prepare Exhibit 3 and submit for BCC approval."
                ),
            )
        )

    if contract_type == "emergency":
        issues.append(
            ComplianceIssue(
                rule_id="DOC-002",
                severity="critical",
                message=(
                    "Emergency procurement requires Emergency Procurement "
                    "Justification (Exhibit 1) signed by division manager."
                ),
                section_reference="Section 4, Exhibit 1",
                suggested_action=(
                    "Complete Exhibit 1. Obtain quotes or explain why quotes "
                    "could not be obtained. Email ProcurementEmergency@ocfl.net. "
                    "If over $100K, Price Negotiation Memo and Board approval "
                    "may be required."
                ),
            )
        )

    # Expediting form for dept quotes up to $100K
    if (
        not is_federal
        and 10_000 < amount <= 100_000
        and contract_type in ("standard", "piggyback", "cooperative")
    ):
        issues.append(
            ComplianceIssue(
                rule_id="DOC-003",
                severity="warning",
                message=(
                    "Department/Division Expedited Quoting Form (Exhibit 40) "
                    "is MANDATORY for competitive quotes obtained at "
                    "department/division level up to $100,000."
                ),
                section_reference="Section 2, Exhibit 40",
                suggested_action=(
                    "Complete Exhibit 40 with recommended source, M/WBE sources "
                    "solicited, and additional sources. County Funded Only — "
                    "disallowed for grants."
                ),
            )
        )

    # COI for services
    if data.scope_of_services and amount > 0:
        scope_lower = data.scope_of_services.lower()
        if any(
            kw in scope_lower
            for kw in ["service", "consulting", "professional", "maintenance"]
        ):
            issues.append(
                ComplianceIssue(
                    rule_id="DOC-004",
                    severity="info",
                    message=(
                        "Services detected in scope. Certificate of Insurance "
                        "(COI) is typically required for service contracts."
                    ),
                    section_reference="Section 10, Exhibit 33",
                    suggested_action=(
                        "Ensure vendor provides COI with appropriate coverage: "
                        "Commercial General Liability ($500K min), Business Auto "
                        "($500K), Workers' Comp (statutory). OC BCC as Additional "
                        "Insured."
                    ),
                )
            )

    # ---- Anti-splitting warning ----
    # Check if there are recent submissions to the same vendor
    if data.vendor_name:
        db = _load_db()
        vendor_total = amount
        vendor_submissions = []
        for sid, sub in db.get("submissions", {}).items():
            sub_extracted = sub.get("extracted_data", {})
            if (
                sub_extracted.get("vendor_name", "").strip().lower()
                == data.vendor_name.strip().lower()
            ):
                vendor_total += sub_extracted.get("total_amount", 0)
                vendor_submissions.append(sid)

        # Determine relevant threshold
        split_threshold = 250_000 if is_federal else 150_000
        if amount <= split_threshold and vendor_total > split_threshold:
            issues.append(
                ComplianceIssue(
                    rule_id="SPLIT-001",
                    severity="critical",
                    message=(
                        f"Anti-splitting alert: This submission (${amount:,.2f}) "
                        f"plus {len(vendor_submissions)} other submission(s) to "
                        f"{data.vendor_name} total ${vendor_total:,.2f}, which "
                        f"exceeds the ${split_threshold:,} threshold. "
                        "Subdividing purchases to avoid bid requirements is "
                        "prohibited."
                    ),
                    section_reference="Section 2, Validation Rules",
                    suggested_action=(
                        "Review all recent purchases to this vendor. If related, "
                        "combine into a single procurement at the appropriate "
                        "threshold level."
                    ),
                )
            )

    return ComplianceResult(
        issues=issues,
        threshold_category=category,
        board_approval_required=board_approval,
        procurement_method=method,
    )


# ---------------------------------------------------------------------------
# Required Forms Determination
# ---------------------------------------------------------------------------


def determine_required_forms(
    data: ExtractedData, funding_type: str = "", request_type: str = ""
) -> list[dict]:
    """Determine which OCFL procurement forms/exhibits are required.

    Returns a list of dicts with: exhibit_number, title, status, reason.
    Status is one of: "required", "recommended", "not_needed".
    """
    amount = data.total_amount
    ft = funding_type or data.funding_type or "standard"
    ct = request_type or data.contract_type or "standard"
    is_federal = ft == "federal"
    forms: list[dict] = []

    # --- Exhibit 1: Emergency Procurement Justification ---
    if ct == "emergency":
        forms.append({
            "exhibit_number": "Exhibit 1",
            "title": "Emergency Procurement Justification",
            "status": "required",
            "reason": "Emergency procurement requires signed justification from division manager.",
        })
    else:
        forms.append({
            "exhibit_number": "Exhibit 1",
            "title": "Emergency Procurement Justification",
            "status": "not_needed",
            "reason": "Not an emergency procurement.",
        })

    # --- Exhibit 2: Sole Source Procurement Data Sheet ---
    if ct == "sole_source":
        forms.append({
            "exhibit_number": "Exhibit 2",
            "title": "Sole Source Procurement Data Sheet",
            "status": "required",
            "reason": "Sole source procurement requires justification data sheet.",
        })
    else:
        forms.append({
            "exhibit_number": "Exhibit 2",
            "title": "Sole Source Procurement Data Sheet",
            "status": "not_needed",
            "reason": "Not a sole source procurement.",
        })

    # --- Exhibit 3: Price Negotiation Memorandum ---
    if ct == "sole_source":
        forms.append({
            "exhibit_number": "Exhibit 3",
            "title": "Price Negotiation Memorandum",
            "status": "required",
            "reason": "Sole source requires price negotiation documentation.",
        })
    elif is_federal and amount > 10_000:
        forms.append({
            "exhibit_number": "Exhibit 3",
            "title": "Price Negotiation Memorandum",
            "status": "recommended",
            "reason": "Federal procurement over $10K may require profit negotiation memo if no price competition.",
        })
    else:
        forms.append({
            "exhibit_number": "Exhibit 3",
            "title": "Price Negotiation Memorandum",
            "status": "not_needed",
            "reason": "Price negotiation memo not required for this procurement type.",
        })

    # --- Exhibit 5: Construction Project Information ---
    scope_lower = (data.scope_of_services or "").lower()
    is_construction = any(kw in scope_lower for kw in ["construct", "building", "renovation", "demolition"])
    if is_construction:
        forms.append({
            "exhibit_number": "Exhibit 5",
            "title": "Construction Project Information Sheet",
            "status": "required" if amount > 10_000 else "recommended",
            "reason": "Construction-related scope detected. Project information sheet needed.",
        })
    else:
        forms.append({
            "exhibit_number": "Exhibit 5",
            "title": "Construction Project Information Sheet",
            "status": "not_needed",
            "reason": "No construction scope detected.",
        })

    # --- Exhibit 6: RFP Project Information Sheet ---
    if amount > 150_000 and not is_federal:
        forms.append({
            "exhibit_number": "Exhibit 6",
            "title": "RFP Project Information Sheet",
            "status": "recommended",
            "reason": f"Amount ${amount:,.2f} exceeds $150K — formal solicitation likely requires project info.",
        })
    elif is_federal and amount > 250_000:
        forms.append({
            "exhibit_number": "Exhibit 6",
            "title": "RFP Project Information Sheet",
            "status": "recommended",
            "reason": f"Federal amount ${amount:,.2f} exceeds $250K — formal solicitation likely.",
        })
    else:
        forms.append({
            "exhibit_number": "Exhibit 6",
            "title": "RFP Project Information Sheet",
            "status": "not_needed",
            "reason": "Amount below formal solicitation threshold.",
        })

    # --- Exhibit 30: Exemption from Competition ---
    if ct in ("sole_source", "emergency"):
        forms.append({
            "exhibit_number": "Exhibit 30",
            "title": "Exemption from Competition",
            "status": "recommended",
            "reason": f"{ct.replace('_', ' ').title()} may require exemption documentation.",
        })
    else:
        forms.append({
            "exhibit_number": "Exhibit 30",
            "title": "Exemption from Competition",
            "status": "not_needed",
            "reason": "Competitive procurement — exemption not needed.",
        })

    # --- Exhibit 31: ACS Approval Form ---
    if ct in ("piggyback", "cooperative", "state_contract", "gsa"):
        forms.append({
            "exhibit_number": "Exhibit 31",
            "title": "Alternate Contract Source (ACS) Approval Form",
            "status": "required",
            "reason": "ACS/piggyback/cooperative procurement requires ACS approval.",
        })
    else:
        forms.append({
            "exhibit_number": "Exhibit 31",
            "title": "Alternate Contract Source (ACS) Approval Form",
            "status": "not_needed",
            "reason": "Not using an alternate contract source.",
        })

    # --- Exhibit 32: Piggyback Requisition Checklist ---
    if ct == "piggyback":
        forms.append({
            "exhibit_number": "Exhibit 32",
            "title": "Piggyback Requisition Checklist",
            "status": "required",
            "reason": "Piggyback procurement requires the requisition checklist.",
        })
    else:
        forms.append({
            "exhibit_number": "Exhibit 32",
            "title": "Piggyback Requisition Checklist",
            "status": "not_needed",
            "reason": "Not a piggyback procurement.",
        })

    # --- Exhibit 33: Short Form RFQ ---
    if 10_000 < amount <= 150_000 and ct == "standard" and not is_federal:
        forms.append({
            "exhibit_number": "Exhibit 33",
            "title": "Short Form Request for Quotations (RFQ)",
            "status": "recommended",
            "reason": f"Amount ${amount:,.2f} is in $10K-$150K range — short form RFQ encouraged.",
        })
    else:
        forms.append({
            "exhibit_number": "Exhibit 33",
            "title": "Short Form Request for Quotations (RFQ)",
            "status": "not_needed",
            "reason": "Short form RFQ not applicable for this procurement type/amount.",
        })

    # --- Exhibit 34: Project Information Sheet ---
    if amount > 150_000:
        forms.append({
            "exhibit_number": "Exhibit 34",
            "title": "Project Information Sheet",
            "status": "required",
            "reason": f"Amount ${amount:,.2f} exceeds $150K — project information sheet required for formal solicitation.",
        })
    else:
        forms.append({
            "exhibit_number": "Exhibit 34",
            "title": "Project Information Sheet",
            "status": "not_needed",
            "reason": "Amount below $150K formal solicitation threshold.",
        })

    # --- Exhibit 35: Independent Cost Estimate ---
    if is_federal and amount > 250_000:
        forms.append({
            "exhibit_number": "Exhibit 35",
            "title": "Independent Cost Estimate",
            "status": "required",
            "reason": "Federal procurement over $250K requires independent cost estimate.",
        })
    elif amount > 150_000:
        forms.append({
            "exhibit_number": "Exhibit 35",
            "title": "Independent Cost Estimate",
            "status": "recommended",
            "reason": "Formal solicitation — independent cost estimate recommended.",
        })
    else:
        forms.append({
            "exhibit_number": "Exhibit 35",
            "title": "Independent Cost Estimate",
            "status": "not_needed",
            "reason": "Not required at this procurement level.",
        })

    # --- Exhibit 36: Federal Compliance Documentation ---
    if is_federal:
        forms.append({
            "exhibit_number": "Exhibit 36",
            "title": "Federal Compliance Documentation Form",
            "status": "required",
            "reason": "Federal funding detected — federal compliance documentation mandatory.",
        })
    else:
        forms.append({
            "exhibit_number": "Exhibit 36",
            "title": "Federal Compliance Documentation Form",
            "status": "not_needed",
            "reason": "Not federally funded.",
        })

    # --- Exhibit 40: Dept Expediting Form with M/WBE ---
    if not is_federal and 10_000 < amount <= 100_000 and ct in ("standard", "piggyback", "cooperative"):
        forms.append({
            "exhibit_number": "Exhibit 40",
            "title": "Department/Division Expedited Quoting Form (M/WBE)",
            "status": "required",
            "reason": f"Amount ${amount:,.2f} is $10K-$100K range with department quoting — Exhibit 40 mandatory.",
        })
    else:
        status = "not_needed"
        reason = "Not in the $10K-$100K department quoting range or not applicable."
        if is_federal and 10_000 < amount <= 100_000:
            reason = "Exhibit 40 is county-funded only — disallowed for grants."
        forms.append({
            "exhibit_number": "Exhibit 40",
            "title": "Department/Division Expedited Quoting Form (M/WBE)",
            "status": status,
            "reason": reason,
        })

    # --- Exhibit 41: Direct Award M/WBE ---
    if ct == "standard" and amount <= 10_000:
        forms.append({
            "exhibit_number": "Exhibit 41",
            "title": "Direct Award M/WBE Certification",
            "status": "recommended",
            "reason": "Small purchase under $10K — consider direct award to M/WBE vendor.",
        })
    else:
        forms.append({
            "exhibit_number": "Exhibit 41",
            "title": "Direct Award M/WBE Certification",
            "status": "not_needed",
            "reason": "Direct award M/WBE not applicable at this amount/type.",
        })

    # Sort: required first, then recommended, then not_needed
    status_order = {"required": 0, "recommended": 1, "not_needed": 2}
    forms.sort(key=lambda f: status_order.get(f["status"], 3))

    return forms


# ---------------------------------------------------------------------------
# Contract Search
# ---------------------------------------------------------------------------

DIRECT_SEARCH_LINKS = [
    {
        "source": "NASPO ValuePoint",
        "url_template": "https://www.naspovaluepoint.org/portfolio/?search={query}",
        "has_api": False,
    },
    {
        "source": "FL State Term (FACTS)",
        "url_template": "https://vendor.myfloridamarketplace.com/search/bids/detail/{query}",
        "has_api": False,
    },
    {
        "source": "OMNIA Partners",
        "url_template": "https://www.omniapartners.com/search?q={query}",
        "has_api": False,
    },
    {
        "source": "Sourcewell",
        "url_template": "https://www.sourcewell-mn.gov/contract-search?q={query}",
        "has_api": False,
    },
    {
        "source": "CDW-G",
        "url_template": "https://www.cdwg.com/search/?key={query}",
        "has_api": False,
    },
    {
        "source": "GSA Advantage",
        "url_template": "https://www.gsaadvantage.gov/advantage/s/search.do?q=0:{query}&db=0&searchType=0",
        "has_api": False,
    },
]


async def _search_usaspending(query: str) -> list[dict]:
    """Search USASpending.gov for awarded contracts by vendor/keyword."""
    url = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
    payload = {
        "filters": {
            "keyword": query,
            "award_type_codes": ["A", "B", "C", "D"],
        },
        "fields": [
            "Award ID",
            "Recipient Name",
            "Description",
            "Start Date",
            "End Date",
            "Award Amount",
            "Awarding Agency",
            "Award Type",
        ],
        "limit": 10,
        "page": 1,
        "sort": "Award Amount",
        "order": "desc",
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for row in data.get("results", []):
                results.append({
                    "source": "USASpending.gov",
                    "vendor_name": row.get("Recipient Name", ""),
                    "contract_number": row.get("Award ID", ""),
                    "description": row.get("Description", ""),
                    "amount": row.get("Award Amount"),
                    "start_date": row.get("Start Date", ""),
                    "end_date": row.get("End Date", ""),
                    "agency": row.get("Awarding Agency", ""),
                    "award_type": row.get("Award Type", ""),
                })
            return results
    except Exception as exc:
        logger.warning("USASpending search failed: %s", exc)
        return []


async def _search_sam_gov(vendor_name: str) -> list[dict]:
    """Search SAM.gov Entity API for vendor registration status."""
    if not SAM_API_KEY:
        return []
    url = "https://api.sam.gov/entity-information/v4/entities"
    params = {
        "api_key": SAM_API_KEY,
        "legalBusinessName": vendor_name,
        "registrationStatus": "A",
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for entity in data.get("entityData", []):
                reg = entity.get("entityRegistration", {})
                core = entity.get("coreData", {})
                gen_info = core.get("generalInformation", {})
                results.append({
                    "source": "SAM.gov",
                    "vendor_name": reg.get("legalBusinessName", ""),
                    "uei": reg.get("ueiSAM", ""),
                    "cage_code": reg.get("cageCode", ""),
                    "registration_status": reg.get("registrationStatus", ""),
                    "expiration_date": reg.get("registrationExpirationDate", ""),
                    "entity_type": gen_info.get("entityStructureDesc", ""),
                    "physical_address": core.get("mailingAddress", {}).get("addressLine1", ""),
                })
            return results
    except Exception as exc:
        logger.warning("SAM.gov search failed: %s", exc)
        return []


async def search_contracts(query: str, search_type: str = "vendor") -> dict:
    """Search all contract sources for the given query."""
    usaspending_results = await _search_usaspending(query)
    sam_results = await _search_sam_gov(query) if search_type == "vendor" else []

    # Build direct search links
    from urllib.parse import quote
    encoded = quote(query)
    direct_links = []
    for link in DIRECT_SEARCH_LINKS:
        direct_links.append({
            "source": link["source"],
            "url": link["url_template"].format(query=encoded),
        })

    return {
        "query": query,
        "search_type": search_type,
        "api_results": usaspending_results + sam_results,
        "direct_search_links": direct_links,
        "sources_searched": {
            "usaspending": {"searched": True, "count": len(usaspending_results)},
            "sam_gov": {"searched": bool(SAM_API_KEY), "count": len(sam_results)},
        },
    }


# ---------------------------------------------------------------------------
# Pricing Intelligence Engine
# ---------------------------------------------------------------------------

PRICING_ANALYSIS_PROMPT = """\
You are a procurement pricing analyst for Orange County, Florida. \
Analyze the uploaded quote details and comparable contract data to produce \
a pricing intelligence report.

QUOTE DETAILS:
- Vendor: {vendor_name}
- Product/Service: {scope_of_services}
- Line Items: {line_items_summary}
- Total Amount: ${total_amount:,.2f}

COMPARABLE CONTRACTS FOUND:
{comparable_data}

Return ONLY valid JSON with:
{{
  "search_queries": ["list of 3-5 short product-specific search queries \
that would find comparable government POs, e.g. 'Dell PowerEdge R750 \
government purchase order', 'network switches county contract award'"],
  "product_keywords": ["list of 2-4 concise product keywords for \
cooperative contract searches, e.g. 'Dell PowerEdge R750', 'network switch'"],
  "price_assessment": "Brief assessment of whether the quoted price \
appears competitive, above market, or below market based on available data",
  "savings_opportunities": ["list of specific actionable suggestions, \
e.g. 'NASPO ValuePoint contract for Dell servers may offer 15-20%% discount', \
'Check GSA IT Schedule 70 for comparable pricing'"],
  "confidence": "low|medium|high — based on how much comparable data was found"
}}
"""


async def _search_usaspending_by_product(description: str) -> list[dict]:
    """Search USASpending.gov by product/service description to find comparable government purchases."""
    url = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
    # Use a shorter keyword from the description for better matches
    keywords = description[:200] if description else ""
    payload = {
        "filters": {
            "keyword": keywords,
            "award_type_codes": ["A", "B", "C", "D"],
        },
        "fields": [
            "Award ID",
            "Recipient Name",
            "Description",
            "Start Date",
            "End Date",
            "Award Amount",
            "Awarding Agency",
            "Award Type",
        ],
        "limit": 15,
        "page": 1,
        "sort": "Award Amount",
        "order": "asc",
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for row in data.get("results", []):
                results.append({
                    "source": "USASpending.gov",
                    "vendor_name": row.get("Recipient Name", ""),
                    "contract_number": row.get("Award ID", ""),
                    "description": row.get("Description", ""),
                    "amount": row.get("Award Amount"),
                    "start_date": row.get("Start Date", ""),
                    "end_date": row.get("End Date", ""),
                    "agency": row.get("Awarding Agency", ""),
                    "award_type": row.get("Award Type", ""),
                })
            return results
    except Exception as exc:
        logger.warning("USASpending product search failed: %s", exc)
        return []


async def _search_web_government_pos(queries: list[str]) -> list[dict]:
    """Search the web for government purchase orders and contract awards.

    Uses a simple Google Custom Search–style approach via DuckDuckGo instant
    answer API (free, no key).  Falls back gracefully.
    """
    results = []
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            for query in queries[:5]:  # limit to 5 queries
                search_query = f"{query} site:gov OR site:org government purchase order contract award"
                resp = await client.get(
                    "https://api.duckduckgo.com/",
                    params={"q": search_query, "format": "json", "no_html": "1"},
                    follow_redirects=True,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    # Collect related topics
                    for topic in data.get("RelatedTopics", [])[:3]:
                        if isinstance(topic, dict) and topic.get("FirstURL"):
                            results.append({
                                "source": "Web Search",
                                "title": topic.get("Text", "")[:200],
                                "url": topic.get("FirstURL", ""),
                                "query_used": query,
                            })
                    # Also check abstract
                    if data.get("AbstractURL"):
                        results.append({
                            "source": "Web Search",
                            "title": data.get("Abstract", data.get("Heading", ""))[:200],
                            "url": data["AbstractURL"],
                            "query_used": query,
                        })
    except Exception as exc:
        logger.warning("Web PO search failed: %s", exc)
    return results


def _call_llm_pricing_analysis(
    extracted: "ExtractedData",
    comparable_contracts: list[dict],
) -> dict:
    """Use local LLM to analyze pricing and generate search recommendations."""
    # Build line items summary
    line_items_summary = "; ".join(
        f"{li.description} (qty {li.quantity}, ${li.unit_price:,.2f} ea)"
        for li in extracted.line_items[:10]
    ) if extracted.line_items else "No line items extracted"

    # Build comparable data summary
    if comparable_contracts:
        comparable_lines = []
        for c in comparable_contracts[:10]:
            line = f"- {c.get('vendor_name', 'Unknown')}: ${c.get('amount', 0):,.2f}"
            if c.get('description'):
                line += f" ({c['description'][:100]})"
            if c.get('agency'):
                line += f" — {c['agency']}"
            comparable_lines.append(line)
        comparable_data = "\n".join(comparable_lines)
    else:
        comparable_data = "No comparable contracts found in initial search."

    prompt = PRICING_ANALYSIS_PROMPT.format(
        vendor_name=extracted.vendor_name,
        scope_of_services=extracted.scope_of_services or "Not specified",
        line_items_summary=line_items_summary,
        total_amount=extracted.total_amount,
        comparable_data=comparable_data,
    )

    try:
        response = httpx.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.2, "num_predict": 2048},
            },
            timeout=120.0,
        )
        response.raise_for_status()
        return _parse_llm_response(response.json())
    except Exception as exc:
        logger.warning("LLM pricing analysis failed: %s", exc)
        # Return sensible defaults when LLM is unavailable
        scope = extracted.scope_of_services or ""
        vendor = extracted.vendor_name or ""
        base_keywords = [w for w in (scope + " " + vendor).split() if len(w) > 3][:4]
        return {
            "search_queries": [
                f"{' '.join(base_keywords[:3])} government purchase order",
                f"{vendor} county contract award",
                f"{' '.join(base_keywords[:2])} state contract pricing",
            ],
            "product_keywords": base_keywords[:3] if base_keywords else ["general"],
            "price_assessment": "Unable to assess — AI analysis unavailable. Review comparable contracts manually.",
            "savings_opportunities": [
                "Check cooperative contracts (NASPO, Sourcewell, GSA) for pre-negotiated pricing.",
                "Search USASpending.gov for comparable federal purchases.",
            ],
            "confidence": "low",
        }


async def run_pricing_intelligence(extracted: "ExtractedData") -> dict:
    """Run the full pricing intelligence pipeline.

    Steps:
    1. Search USASpending by product description (not just vendor)
    2. Ask LLM to analyze quote vs comparable data and suggest searches
    3. Run web searches for other government POs
    4. Generate cooperative contract search links
    5. Return structured pricing intelligence report
    """
    import asyncio
    from urllib.parse import quote

    # Step 1: Product-based search on USASpending
    product_query = extracted.scope_of_services or ""
    if not product_query and extracted.line_items:
        product_query = " ".join(
            li.description for li in extracted.line_items[:3]
        )
    if not product_query:
        product_query = extracted.vendor_name

    usaspending_product_results = await _search_usaspending_by_product(product_query)

    # Step 2: LLM analysis — generates search queries + assessment
    llm_analysis = _call_llm_pricing_analysis(extracted, usaspending_product_results)

    # Step 3: Web search for other government POs
    web_search_queries = llm_analysis.get("search_queries", [])
    web_po_results = await _search_web_government_pos(web_search_queries)

    # Step 4: Cooperative contract search links (product-specific)
    product_keywords = llm_analysis.get("product_keywords", [])
    primary_keyword = product_keywords[0] if product_keywords else product_query[:80]
    encoded_kw = quote(primary_keyword)

    coop_search_links = []
    for link_def in DIRECT_SEARCH_LINKS:
        coop_search_links.append({
            "source": link_def["source"],
            "url": link_def["url_template"].format(query=encoded_kw),
            "search_term": primary_keyword,
        })

    # Step 5: Calculate basic stats from comparable contracts
    comparable_amounts = [
        r["amount"] for r in usaspending_product_results
        if r.get("amount") and isinstance(r["amount"], (int, float)) and r["amount"] > 0
    ]
    price_stats = {}
    if comparable_amounts:
        price_stats = {
            "min": min(comparable_amounts),
            "max": max(comparable_amounts),
            "avg": sum(comparable_amounts) / len(comparable_amounts),
            "median": sorted(comparable_amounts)[len(comparable_amounts) // 2],
            "count": len(comparable_amounts),
        }

    return {
        "quote_amount": extracted.total_amount,
        "vendor_name": extracted.vendor_name,
        "product_searched": product_query[:200],
        "comparable_contracts": usaspending_product_results[:10],
        "web_po_results": web_po_results[:10],
        "coop_search_links": coop_search_links,
        "price_stats": price_stats,
        "llm_analysis": {
            "price_assessment": llm_analysis.get("price_assessment", ""),
            "savings_opportunities": llm_analysis.get("savings_opportunities", []),
            "confidence": llm_analysis.get("confidence", "low"),
        },
        "search_queries_used": web_search_queries,
    }


# ---------------------------------------------------------------------------
# PDF generation (ReportLab)
# ---------------------------------------------------------------------------


def _generate_piggyback_checklist_pdf(
    data: ExtractedData, overrides: dict[str, Any], out_path: Path
) -> None:
    """Generate a filled Piggyback Requisition Checklist PDF."""
    merged = data.model_dump()
    merged.update(overrides)

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=14,
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.grey,
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "SectionHead",
        parent=styles["Heading2"],
        fontSize=11,
        spaceAfter=4,
        spaceBefore=12,
    )
    normal = styles["Normal"]

    elements: list[Any] = []

    # Title
    elements.append(Paragraph("Orange County, Florida", title_style))
    elements.append(
        Paragraph("Piggyback Requisition Checklist (Exhibit 32)", subtitle_style)
    )
    elements.append(
        Paragraph(
            "FOR APPROVED ALTERNATE CONTRACT SOURCES (ACS) ONLY", subtitle_style
        )
    )

    # Header table
    header_data = [
        ["Vendor Name:", merged.get("vendor_name", "")],
        ["Requisition Number:", merged.get("requisition_number", "")],
        ["Date Submitted:", merged.get("date_submitted", "")],
        ["Department/Division:", merged.get("department", "")],
        ["Requestor Name:", merged.get("requestor_name", "")],
        ["Requestor Phone:", merged.get("requestor_phone", "")],
        ["Contract Number:", merged.get("contract_number", "")],
    ]
    header_table = Table(header_data, colWidths=[2 * inch, 4.5 * inch])
    header_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("BACKGROUND", (0, 0), (0, -1), colors.Color(0.95, 0.95, 0.95)),
            ]
        )
    )
    elements.append(header_table)
    elements.append(Spacer(1, 12))

    # Steps
    elements.append(Paragraph("Step 1 — Contract Verification", heading_style))
    elements.append(
        Paragraph(
            f"Contract Number: <b>{merged.get('contract_number', 'N/A')}</b>",
            normal,
        )
    )
    elements.append(
        Paragraph("☑ Confirm Contract is on the ACS log", normal)
    )
    exp = merged.get("expiration_date", "")
    exp_status = "NOT EXPIRED" if exp else "Verify expiration"
    elements.append(
        Paragraph(
            f"☑ Contract is not expired (Expiration: {exp or 'N/A'} — {exp_status})",
            normal,
        )
    )

    elements.append(Paragraph("Step 2 — Contract Quotation", heading_style))
    elements.append(
        Paragraph(
            f"☑ Vendor Name on contract matches quote and requisition: "
            f"<b>{merged.get('vendor_name', 'N/A')}</b>",
            normal,
        )
    )

    elements.append(
        Paragraph("Step 3 — Line-by-Line Pricing Validation", heading_style)
    )

    line_items = merged.get("line_items", [])
    if line_items:
        li_header = ["Description", "Qty", "Unit Price", "Extended"]
        li_rows = [li_header]
        for item in line_items:
            if isinstance(item, dict):
                li_rows.append(
                    [
                        str(item.get("description", "")),
                        str(item.get("quantity", "")),
                        f"${item.get('unit_price', 0):,.2f}",
                        f"${item.get('extended_price', 0):,.2f}",
                    ]
                )
        li_rows.append(
            [
                "TOTAL",
                "",
                "",
                f"${merged.get('total_amount', 0):,.2f}",
            ]
        )
        li_table = Table(li_rows, colWidths=[3 * inch, 0.75 * inch, 1.25 * inch, 1.25 * inch])
        li_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.9, 0.9, 0.9)),
                    ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                    ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                ]
            )
        )
        elements.append(li_table)
    else:
        elements.append(
            Paragraph("<i>No line items extracted. Attach price sheets.</i>", normal)
        )

    # Scope
    elements.append(Paragraph("Scope of Services", heading_style))
    elements.append(
        Paragraph(merged.get("scope_of_services", "N/A"), normal)
    )

    # Signature lines
    elements.append(Spacer(1, 30))
    sig_data = [
        ["Buyer Signature:", "________________________", "Date:", "____________"],
        [
            "Procurement Manager:",
            "________________________",
            "Date:",
            "____________",
        ],
    ]
    sig_table = Table(
        sig_data, colWidths=[1.5 * inch, 2.5 * inch, 0.6 * inch, 1.5 * inch]
    )
    sig_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ]
        )
    )
    elements.append(sig_table)

    # Footer
    elements.append(Spacer(1, 12))
    elements.append(
        Paragraph(
            f"<i>Generated by OCFL Procurement Portal — "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}</i>",
            ParagraphStyle("Footer", parent=normal, fontSize=8, textColor=colors.grey),
        )
    )

    doc.build(elements)


def _generate_expediting_form_pdf(
    data: ExtractedData, overrides: dict[str, Any], out_path: Path
) -> None:
    """Generate a filled Department Expediting Form (Exhibit 40) PDF."""
    merged = data.model_dump()
    merged.update(overrides)

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Title"], fontSize=14, spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.grey,
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "SectionHead",
        parent=styles["Heading2"],
        fontSize=11,
        spaceAfter=4,
        spaceBefore=12,
    )
    normal = styles["Normal"]

    elements: list[Any] = []

    elements.append(Paragraph("Orange County, Florida", title_style))
    elements.append(
        Paragraph(
            "Department/Division Expedited Quoting Form (Exhibit 40)", subtitle_style
        )
    )
    elements.append(
        Paragraph(
            "Up to $100,000 — County Funded Only — DISALLOWED FOR GRANTS",
            subtitle_style,
        )
    )

    header_data = [
        ["Date of Request:", merged.get("date_submitted", "")],
        ["Department/Division:", merged.get("department", "")],
        ["Requisition Number:", merged.get("requisition_number", "")],
        ["Requestor Name:", merged.get("requestor_name", "")],
        ["Amount of Purchase:", f"${merged.get('total_amount', 0):,.2f}"],
        ["Requestor Phone:", merged.get("requestor_phone", "")],
    ]
    header_table = Table(header_data, colWidths=[2 * inch, 4.5 * inch])
    header_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("BACKGROUND", (0, 0), (0, -1), colors.Color(0.95, 0.95, 0.95)),
            ]
        )
    )
    elements.append(header_table)

    elements.append(Paragraph("Recommended Source (Quote Attached)", heading_style))
    rec_data = [
        ["Vendor/Supplier", "Amount", "M/WBE Status"],
        [merged.get("vendor_name", ""), f"${merged.get('total_amount', 0):,.2f}", "☐ M/WBE  ☐ Non-M/WBE"],
    ]
    rec_table = Table(rec_data, colWidths=[3 * inch, 1.5 * inch, 2 * inch])
    rec_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.9, 0.9, 0.9)),
            ]
        )
    )
    elements.append(rec_table)

    elements.append(
        Paragraph("M/WBE Sources Solicited (Quotes Attached)", heading_style)
    )
    mwbe_data = [
        ["Vendor/Supplier", "Amount", "No Answer"],
        ["", "", "☐"],
        ["", "", "☐"],
    ]
    mwbe_table = Table(mwbe_data, colWidths=[3 * inch, 1.5 * inch, 2 * inch])
    mwbe_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.9, 0.9, 0.9)),
            ]
        )
    )
    elements.append(mwbe_table)

    elements.append(
        Paragraph("Additional Sources Solicited (Quotes Attached)", heading_style)
    )
    add_data = [
        ["Vendor/Supplier", "Amount", "No Answer"],
        ["", "", "☐"],
        ["", "", "☐"],
    ]
    add_table = Table(add_data, colWidths=[3 * inch, 1.5 * inch, 2 * inch])
    add_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.9, 0.9, 0.9)),
            ]
        )
    )
    elements.append(add_table)

    # Signature
    elements.append(Spacer(1, 24))
    elements.append(
        Paragraph(
            "Requestor Signature: ________________________  Date: ____________",
            normal,
        )
    )
    elements.append(Spacer(1, 12))
    elements.append(
        Paragraph(
            "Division Manager Signature: ________________________  Date: ____________",
            normal,
        )
    )

    elements.append(Spacer(1, 12))
    elements.append(
        Paragraph(
            f"<i>Generated by OCFL Procurement Portal — "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}</i>",
            ParagraphStyle("Footer", parent=normal, fontSize=8, textColor=colors.grey),
        )
    )

    doc.build(elements)


# ---------------------------------------------------------------------------
# Approval Routing Engine
# ---------------------------------------------------------------------------


def determine_approval_route(data: ExtractedData, compliance: ComplianceResult) -> ApprovalRoute:
    """Determine where a submission should be routed based on OCFL thresholds."""
    amount = data.total_amount
    is_federal = data.funding_type == "federal"

    if amount <= 10_000:
        return ApprovalRoute(
            route_to="supervisor",
            reason=f"Amount ${amount:,.2f} is within small purchase threshold ($0\u2013$10K). Supervisor approval sufficient.",
            threshold_category="small_purchase",
            required_approvals=["Division Supervisor"],
            estimated_lead_time="1\u20132 business days",
        )
    elif amount <= 100_000 and not is_federal:
        return ApprovalRoute(
            route_to="supervisor_plus_procurement",
            reason=f"Amount ${amount:,.2f} requires department-level quoting ($10K\u2013$100K). Division supervisor approval + Procurement review.",
            threshold_category="department_quotes",
            required_approvals=["Division Supervisor", "Procurement Analyst"],
            estimated_lead_time="3\u20135 business days",
        )
    elif amount <= 150_000 and not is_federal:
        return ApprovalRoute(
            route_to="procurement",
            reason=f"Amount ${amount:,.2f} requires informal quotes ($100K\u2013$150K). Procurement Division handles solicitation.",
            threshold_category="informal_quotes",
            required_approvals=["Division Manager", "Procurement Analyst", "Procurement Manager"],
            estimated_lead_time="2\u20134 weeks",
        )
    elif amount <= 500_000:
        return ApprovalRoute(
            route_to="procurement_formal",
            reason=f"Amount ${amount:,.2f} exceeds $150K. Formal sealed solicitation required by Procurement Division.",
            threshold_category="formal_solicitation",
            required_approvals=["Division Manager", "Department Director", "Procurement Manager", "Procurement Division Chief"],
            estimated_lead_time="6\u20138 weeks",
        )
    else:
        return ApprovalRoute(
            route_to="bcc",
            reason=f"Amount ${amount:,.2f} exceeds $500K. Board of County Commissioners approval required.",
            threshold_category="bcc_approval",
            required_approvals=["Division Manager", "Department Director", "Procurement Division", "County Administrator", "Board of County Commissioners"],
            estimated_lead_time="10\u201312 weeks",
        )


def _add_audit_event(record: dict, event_type: str, description: str, actor: str = "system", metadata: dict = None):
    """Add an audit trail event to a submission record."""
    if "audit_trail" not in record:
        record["audit_trail"] = []
    record["audit_trail"].append({
        "event_type": event_type,
        "timestamp": datetime.now().isoformat(),
        "actor": actor,
        "description": description,
        "metadata": metadata or {},
    })


# ---------------------------------------------------------------------------
# Duplicate / Related Purchase Detection
# ---------------------------------------------------------------------------


def detect_duplicates(data: ExtractedData, submission_id: str = "") -> list[dict]:
    """Detect potential duplicate or related purchases."""
    from datetime import timedelta
    db = _load_db()
    alerts = []
    vendor_name = (data.vendor_name or "").strip().lower()
    scope = (data.scope_of_services or "").strip().lower()
    amount = data.total_amount

    if not vendor_name:
        return alerts

    vendor_matches = []
    ninety_days_ago = (datetime.now() - timedelta(days=90)).isoformat()[:10]

    for sid, sub in db.get("submissions", {}).items():
        if sid == submission_id:
            continue
        sub_ed = sub.get("extracted_data", {})
        sub_vendor = (sub_ed.get("vendor_name", "") or "").strip().lower()
        sub_uploaded = sub.get("uploaded_at", "")

        if sub_vendor == vendor_name:
            vendor_matches.append({
                "submission_id": sid,
                "vendor_name": sub_ed.get("vendor_name", ""),
                "amount": sub_ed.get("total_amount", 0),
                "scope": (sub_ed.get("scope_of_services", "") or "")[:100],
                "date": sub_uploaded[:10] if sub_uploaded else "",
                "status": sub.get("status", ""),
            })

    recent_vendor = [m for m in vendor_matches if m.get("date", "") >= ninety_days_ago]
    if recent_vendor:
        combined = amount + sum(m["amount"] for m in recent_vendor)
        is_federal = data.funding_type == "federal"
        split_threshold = 250_000 if is_federal else 150_000

        severity = "warning" if combined > 50_000 else "info"
        if combined > split_threshold and amount <= split_threshold:
            severity = "critical"

        alerts.append({
            "alert_type": "consolidation_opportunity",
            "severity": severity,
            "message": (
                f"{len(recent_vendor)} other purchase(s) to {data.vendor_name} in the past 90 days "
                f"totaling ${sum(m['amount'] for m in recent_vendor):,.2f}. "
                f"Combined with this request: ${combined:,.2f}."
            ),
            "related_submissions": recent_vendor,
            "combined_total": combined,
            "suggested_action": (
                "Consider consolidating into a single purchase for better pricing."
                + (f" Combined total exceeds ${split_threshold:,} threshold \u2014 higher procurement method may be required." if combined > split_threshold else "")
            ),
        })

    return alerts


# ---------------------------------------------------------------------------
# Vendor Scorecard Engine
# ---------------------------------------------------------------------------


def build_vendor_scorecard(vendor_name: str) -> dict:
    """Build a performance scorecard for a vendor."""
    db = _load_db()
    vendor_lower = vendor_name.strip().lower()

    submissions = []
    for sid, sub in db.get("submissions", {}).items():
        ed = sub.get("extracted_data", {})
        if (ed.get("vendor_name", "") or "").strip().lower() == vendor_lower:
            submissions.append(sub)

    if not submissions:
        return {"vendor_name": vendor_name, "found": False}

    amounts = [s.get("extracted_data", {}).get("total_amount", 0) for s in submissions]
    dates = [s.get("uploaded_at", "")[:10] for s in submissions if s.get("uploaded_at")]
    types = list(set(
        s.get("extracted_data", {}).get("type_of_request", "")
        for s in submissions if s.get("extracted_data", {}).get("type_of_request")
    ))

    total_issues = 0
    critical_issues = 0
    warning_issues = 0
    for s in submissions:
        issues = s.get("compliance", {}).get("issues", [])
        total_issues += len(issues)
        critical_issues += len([i for i in issues if i.get("severity") == "critical"])
        warning_issues += len([i for i in issues if i.get("severity") == "warning"])

    status_counts = {}
    for s in submissions:
        st = s.get("status", "pending_review")
        status_counts[st] = status_counts.get(st, 0) + 1

    return {
        "vendor_name": vendor_name,
        "found": True,
        "total_purchases": len(submissions),
        "total_spend": sum(amounts),
        "avg_amount": sum(amounts) / len(amounts) if amounts else 0,
        "min_amount": min(amounts) if amounts else 0,
        "max_amount": max(amounts) if amounts else 0,
        "first_purchase": min(dates) if dates else "",
        "last_purchase": max(dates) if dates else "",
        "types_of_request": types,
        "compliance_summary": {"total_issues": total_issues, "critical": critical_issues, "warnings": warning_issues},
        "status_distribution": status_counts,
        "submissions": [
            {
                "submission_id": s.get("submission_id", ""),
                "date": s.get("uploaded_at", "")[:10] if s.get("uploaded_at") else "",
                "amount": s.get("extracted_data", {}).get("total_amount", 0),
                "scope": (s.get("extracted_data", {}).get("scope_of_services", "") or "")[:100],
                "status": s.get("status", ""),
            }
            for s in sorted(submissions, key=lambda x: x.get("uploaded_at", ""), reverse=True)
        ],
    }


def build_all_vendor_scorecards() -> list[dict]:
    """Build scorecards for all vendors."""
    db = _load_db()
    vendors = set()
    for sid, sub in db.get("submissions", {}).items():
        vn = (sub.get("extracted_data", {}).get("vendor_name", "") or "").strip()
        if vn:
            vendors.add(vn)
    return [sc for v in sorted(vendors) if (sc := build_vendor_scorecard(v)).get("found")]


# ---------------------------------------------------------------------------
# Smart Form Auto-Fill
# ---------------------------------------------------------------------------


def get_vendor_autofill(vendor_name: str) -> dict:
    """Look up previous submissions for a vendor and return auto-fill data."""
    db = _load_db()
    vendor_lower = vendor_name.strip().lower()

    matches = []
    for sid, sub in db.get("submissions", {}).items():
        ed = sub.get("extracted_data", {})
        if (ed.get("vendor_name", "") or "").strip().lower() == vendor_lower:
            matches.append(sub)

    if not matches:
        return {"found": False, "vendor_name": vendor_name}

    matches.sort(key=lambda x: x.get("uploaded_at", ""), reverse=True)
    latest = matches[0].get("extracted_data", {})

    return {
        "found": True,
        "vendor_name": vendor_name,
        "previous_purchases": len(matches),
        "suggested_fields": {
            "vendor_number": latest.get("vendor_number", ""),
            "department": latest.get("department", ""),
            "accounting_line": latest.get("accounting_line", ""),
            "type_of_request": latest.get("type_of_request", ""),
            "contract_number": latest.get("contract_number", ""),
            "contract_type": latest.get("contract_type", ""),
            "requestor_name": latest.get("requestor_name", ""),
            "custodian_code_shipping_notes": latest.get("custodian_code_shipping_notes", ""),
        },
        "last_amount": latest.get("total_amount", 0),
        "last_purchase_date": matches[0].get("uploaded_at", "")[:10],
    }


# ---------------------------------------------------------------------------
# Expiration / Renewal Alerts Engine
# ---------------------------------------------------------------------------


def get_expiration_alerts() -> list[dict]:
    """Check all submissions for upcoming contract/quote expirations."""
    db = _load_db()
    alerts = []
    today = date.today()

    for sid, sub in db.get("submissions", {}).items():
        ed = sub.get("extracted_data", {})
        exp_date_str = ed.get("expiration_date", "")
        if not exp_date_str:
            continue
        try:
            exp_date = datetime.strptime(exp_date_str, "%Y-%m-%d").date()
        except ValueError:
            continue

        days_until = (exp_date - today).days
        vendor = ed.get("vendor_name", "Unknown")
        amount = ed.get("total_amount", 0)
        scope = (ed.get("scope_of_services", "") or "")[:100]

        if days_until < 0:
            alerts.append({"submission_id": sid, "vendor_name": vendor, "expiration_date": exp_date_str, "days_until_expiration": days_until, "severity": "critical", "message": f"EXPIRED {abs(days_until)} days ago", "amount": amount, "scope": scope})
        elif days_until <= 30:
            alerts.append({"submission_id": sid, "vendor_name": vendor, "expiration_date": exp_date_str, "days_until_expiration": days_until, "severity": "critical", "message": f"Expires in {days_until} days", "amount": amount, "scope": scope})
        elif days_until <= 60:
            alerts.append({"submission_id": sid, "vendor_name": vendor, "expiration_date": exp_date_str, "days_until_expiration": days_until, "severity": "warning", "message": f"Expires in {days_until} days", "amount": amount, "scope": scope})
        elif days_until <= 90:
            alerts.append({"submission_id": sid, "vendor_name": vendor, "expiration_date": exp_date_str, "days_until_expiration": days_until, "severity": "info", "message": f"Expires in {days_until} days", "amount": amount, "scope": scope})

    alerts.sort(key=lambda a: a["days_until_expiration"])
    return alerts


# ---------------------------------------------------------------------------
# Spend Analytics Engine
# ---------------------------------------------------------------------------


def compute_spend_analytics() -> dict:
    """Compute comprehensive spend analytics from all submissions."""
    db = _load_db()
    submissions = list(db.get("submissions", {}).values())

    if not submissions:
        return {"total_submissions": 0}

    total_amount = 0
    by_vendor = {}
    by_type = {}
    by_department = {}
    by_month = {}
    by_status = {}
    by_threshold = {}
    amounts = []

    for s in submissions:
        ed = s.get("extracted_data", {})
        comp = s.get("compliance", {})
        amount = ed.get("total_amount", 0) or 0
        total_amount += amount
        amounts.append(amount)

        vendor = (ed.get("vendor_name", "") or "Unknown").strip()
        if vendor:
            by_vendor.setdefault(vendor, {"count": 0, "total": 0})
            by_vendor[vendor]["count"] += 1
            by_vendor[vendor]["total"] += amount

        req_type = ed.get("type_of_request", "") or "Unclassified"
        by_type.setdefault(req_type, {"count": 0, "total": 0})
        by_type[req_type]["count"] += 1
        by_type[req_type]["total"] += amount

        dept = (ed.get("department", "") or "Unknown").strip()
        by_department.setdefault(dept, {"count": 0, "total": 0})
        by_department[dept]["count"] += 1
        by_department[dept]["total"] += amount

        uploaded = s.get("uploaded_at", "")
        if uploaded:
            month_key = uploaded[:7]
            by_month.setdefault(month_key, {"count": 0, "total": 0})
            by_month[month_key]["count"] += 1
            by_month[month_key]["total"] += amount

        status = s.get("status", "pending_review")
        by_status.setdefault(status, {"count": 0, "total": 0})
        by_status[status]["count"] += 1
        by_status[status]["total"] += amount

        threshold = comp.get("threshold_category", "unknown")
        by_threshold.setdefault(threshold, {"count": 0, "total": 0})
        by_threshold[threshold]["count"] += 1
        by_threshold[threshold]["total"] += amount

    top_vendors = sorted(by_vendor.items(), key=lambda x: x[1]["total"], reverse=True)[:10]

    return {
        "total_submissions": len(submissions),
        "total_spend": total_amount,
        "avg_purchase": total_amount / len(submissions) if submissions else 0,
        "min_purchase": min(amounts) if amounts else 0,
        "max_purchase": max(amounts) if amounts else 0,
        "by_vendor": [{"vendor": k, **v} for k, v in top_vendors],
        "by_type": [{"type": k, **v} for k, v in sorted(by_type.items(), key=lambda x: x[1]["total"], reverse=True)],
        "by_department": [{"department": k, **v} for k, v in sorted(by_department.items(), key=lambda x: x[1]["total"], reverse=True)],
        "by_month": [{"month": k, **v} for k, v in sorted(by_month.items())],
        "by_status": [{"status": k, **v} for k, v in by_status.items()],
        "by_threshold": [{"threshold": k, **v} for k, v in by_threshold.items()],
        "unique_vendors": len(by_vendor),
        "unique_departments": len(by_department),
    }


# ---------------------------------------------------------------------------
# Natural Language Query Engine
# ---------------------------------------------------------------------------

NLQ_PROMPT = """\
You are a data analyst for Orange County, Florida's ISS procurement portal. \
Answer the user's question based on the data provided.

SUBMISSION DATA (JSON):
{submissions_summary}

USER QUESTION: {question}

Rules:
- Answer concisely and specifically with numbers when possible
- Format dollar amounts with commas and $ symbol
- If the data doesn't contain enough info, say so
- Reference specific vendors, dates, and amounts when relevant

Return ONLY valid JSON:
{{
  "answer": "Your natural language answer",
  "data": [],
  "query_type": "spend|vendor|status|compliance|general",
  "confidence": "low|medium|high"
}}
"""


def _build_submissions_summary() -> str:
    """Build a compact summary of all submissions for NLQ context."""
    db = _load_db()
    summaries = []
    for sid, sub in db.get("submissions", {}).items():
        ed = sub.get("extracted_data", {})
        comp = sub.get("compliance", {})
        summaries.append({
            "id": sid[:8],
            "date": (sub.get("uploaded_at", "") or "")[:10],
            "vendor": ed.get("vendor_name", ""),
            "amount": ed.get("total_amount", 0),
            "scope": (ed.get("scope_of_services", "") or "")[:80],
            "department": ed.get("department", ""),
            "requestor": ed.get("requestor_name", ""),
            "type": ed.get("type_of_request", ""),
            "contract_type": ed.get("contract_type", ""),
            "status": sub.get("status", ""),
            "threshold": comp.get("threshold_category", ""),
            "funding_type": ed.get("funding_type", "standard"),
            "expiration": ed.get("expiration_date", ""),
        })
    return json.dumps(summaries, indent=1)


def run_natural_language_query(question: str) -> dict:
    """Run a natural language query against submission data using local LLM."""
    summary = _build_submissions_summary()
    prompt = NLQ_PROMPT.format(submissions_summary=summary, question=question)

    try:
        response = httpx.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1, "num_predict": 2048},
            },
            timeout=120.0,
        )
        response.raise_for_status()
        result = _parse_llm_response(response.json())
        return {
            "question": question,
            "answer": result.get("answer", "Unable to process query."),
            "data": result.get("data", []),
            "query_type": result.get("query_type", "general"),
            "confidence": result.get("confidence", "medium"),
        }
    except Exception:
        return _fallback_nlq(question)


def _fallback_nlq(question: str) -> dict:
    """Fallback NLQ when Ollama is unavailable."""
    db = _load_db()
    submissions = list(db.get("submissions", {}).values())
    q = question.lower()

    if any(kw in q for kw in ["spend", "spent", "total", "how much", "cost", "amount"]):
        for s in submissions:
            vendor = (s.get("extracted_data", {}).get("vendor_name", "") or "").lower()
            if vendor and vendor in q:
                vendor_subs = [sub for sub in submissions if (sub.get("extracted_data", {}).get("vendor_name", "") or "").lower() == vendor]
                total = sum(sub.get("extracted_data", {}).get("total_amount", 0) for sub in vendor_subs)
                return {"question": question, "answer": f"Total spend with {s.get('extracted_data', {}).get('vendor_name', '')}: ${total:,.2f} across {len(vendor_subs)} purchase(s).", "data": [], "query_type": "spend", "confidence": "high"}
        total = sum(s.get("extracted_data", {}).get("total_amount", 0) for s in submissions)
        return {"question": question, "answer": f"Total spend across all {len(submissions)} submissions: ${total:,.2f}.", "data": [], "query_type": "spend", "confidence": "high"}

    if any(kw in q for kw in ["pending", "approved", "rejected", "status"]):
        status_counts = {}
        for s in submissions:
            st = s.get("status", "pending_review")
            status_counts[st] = status_counts.get(st, 0) + 1
        parts = [f"{v} {k.replace('_', ' ')}" for k, v in status_counts.items()]
        return {"question": question, "answer": f"Current statuses: {', '.join(parts)}.", "data": [], "query_type": "status", "confidence": "high"}

    if any(kw in q for kw in ["vendor", "supplier"]):
        vendors = {}
        for s in submissions:
            v = (s.get("extracted_data", {}).get("vendor_name", "") or "").strip()
            if v:
                vendors.setdefault(v, {"count": 0, "total": 0})
                vendors[v]["count"] += 1
                vendors[v]["total"] += s.get("extracted_data", {}).get("total_amount", 0)
        top = sorted(vendors.items(), key=lambda x: x[1]["total"], reverse=True)[:5]
        parts = [f"{v}: ${d['total']:,.2f} ({d['count']} purchases)" for v, d in top]
        return {"question": question, "answer": f"Top vendors by spend: " + "; ".join(parts), "data": [], "query_type": "vendor", "confidence": "medium"}

    total = sum(s.get("extracted_data", {}).get("total_amount", 0) for s in submissions)
    return {"question": question, "answer": f"I found {len(submissions)} submissions with ${total:,.2f} in total spend. Try asking about spending by vendor, statuses, or specific amounts.", "data": [], "query_type": "general", "confidence": "low"}


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a vendor quote PDF or image and extract procurement fields."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = Path(file.filename).suffix.lower()
    allowed_exts = {".pdf", ".jpg", ".jpeg", ".png"}
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed_exts)}",
        )

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds maximum size of {MAX_UPLOAD_BYTES // (1024*1024)} MB",
        )

    submission_id = str(uuid.uuid4())
    safe_filename = f"{submission_id}{ext}"
    file_path = UPLOADS_DIR / safe_filename
    file_path.write_bytes(contents)

    # Extract data via AI
    try:
        if ext == ".pdf":
            text = _extract_text_from_pdf(file_path)
            if not text.strip():
                # Fallback: if no text extracted, treat first page as image
                raw_data = _call_llm_text(
                    "(No extractable text found in PDF. "
                    "The document may be a scanned image.)"
                )
            else:
                raw_data = _call_llm_text(text)
        else:
            media_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
            }
            raw_data = _call_llm_image(file_path, media_map[ext])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"AI extraction failed: {exc}",
        )

    # Normalize line_items
    raw_items = raw_data.get("line_items", [])
    normalized_items = []
    for item in raw_items:
        if isinstance(item, dict):
            normalized_items.append(
                {
                    "description": str(item.get("description", "")),
                    "quantity": item.get("quantity", 0),
                    "unit_price": item.get("unit_price", 0),
                    "extended_price": item.get("extended_price", 0),
                }
            )
    raw_data["line_items"] = normalized_items

    extracted = ExtractedData(**{
        k: v for k, v in raw_data.items() if k in ExtractedData.model_fields
    })

    compliance = run_compliance_checks(extracted)
    required_forms = determine_required_forms(extracted)

    # Auto-check contracts if vendor name found
    contract_matches = {}
    if extracted.vendor_name:
        try:
            contract_matches = await search_contracts(extracted.vendor_name, "vendor")
        except Exception as exc:
            logger.warning("Auto contract search failed: %s", exc)
            contract_matches = {"query": extracted.vendor_name, "api_results": [], "error": str(exc)}

    # Run Pricing Intelligence — searches for better pricing on
    # cooperative contracts and comparable POs from other agencies
    pricing_intelligence = {}
    try:
        pricing_intelligence = await run_pricing_intelligence(extracted)
    except Exception as exc:
        logger.warning("Pricing intelligence failed: %s", exc)
        pricing_intelligence = {"error": str(exc)}

    # Persist
    db = _load_db()
    # Determine initial status from compliance results
    critical_issues = [i for i in compliance.issues if i.severity == "critical"]
    initial_status = "pending_review" if critical_issues else "pending_review"

    # Compute approval routing
    route = determine_approval_route(extracted, compliance)
    # Detect duplicates
    duplicate_alerts = detect_duplicates(extracted, submission_id)

    record = {
        "submission_id": submission_id,
        "filename": file.filename,
        "saved_as": safe_filename,
        "uploaded_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "status": initial_status,
        "extracted_data": extracted.model_dump(),
        "compliance": compliance.model_dump(),
        "required_forms": required_forms,
        "contract_matches": contract_matches,
        "pricing_intelligence": pricing_intelligence,
        "approval_routing": route.model_dump(),
        "duplicate_alerts": duplicate_alerts,
        "audit_trail": [],
        "notes": [],
    }
    # Build initial audit trail
    _add_audit_event(record, "created", f"Document '{file.filename}' uploaded and processed. AI extracted {len(extracted.line_items)} line items, total ${extracted.total_amount:,.2f}.")
    _add_audit_event(record, "compliance_check", f"Compliance check completed: {len([i for i in compliance.issues if i.severity == 'critical'])} critical, {len([i for i in compliance.issues if i.severity == 'warning'])} warnings.")
    _add_audit_event(record, "routing", f"Routed to: {route.route_to.replace('_', ' ').title()}. Estimated lead time: {route.estimated_lead_time}.", metadata={"route": route.model_dump()})
    if duplicate_alerts:
        _add_audit_event(record, "duplicate_alert", f"{len(duplicate_alerts)} duplicate/related purchase alert(s) detected.", metadata={"alerts": duplicate_alerts})

    db["submissions"][submission_id] = record
    _save_db(db)

    return {
        "submission_id": submission_id,
        "filename": file.filename,
        "uploaded_at": record["uploaded_at"],
        "extracted_data": extracted.model_dump(),
        "compliance": compliance.model_dump(),
        "required_forms": required_forms,
        "contract_matches": contract_matches,
        "pricing_intelligence": pricing_intelligence,
        "approval_routing": route.model_dump(),
        "duplicate_alerts": duplicate_alerts,
    }


@app.post("/api/compliance-check", response_model=ComplianceResult)
async def compliance_check(req: ComplianceCheckRequest):
    """Re-run compliance checks on user-edited extracted data."""
    db = _load_db()
    if req.submission_id not in db.get("submissions", {}):
        raise HTTPException(status_code=404, detail="Submission not found")

    compliance = run_compliance_checks(req.extracted_data)

    # Update stored data
    db["submissions"][req.submission_id]["extracted_data"] = req.extracted_data.model_dump()
    db["submissions"][req.submission_id]["compliance"] = compliance.model_dump()
    _save_db(db)

    return compliance


@app.get("/api/submission/{submission_id}")
async def get_submission(submission_id: str):
    """Get submission details by ID."""
    db = _load_db()
    record = db.get("submissions", {}).get(submission_id)
    if not record:
        raise HTTPException(status_code=404, detail="Submission not found")
    return record


@app.get("/api/submissions")
async def list_submissions():
    """List all submissions, most recent first."""
    db = _load_db()
    submissions = list(db.get("submissions", {}).values())
    submissions.sort(key=lambda s: s.get("uploaded_at", ""), reverse=True)
    return {"submissions": submissions, "total": len(submissions)}


@app.patch("/api/submissions/{submission_id}/status")
async def update_submission_status(submission_id: str):
    """Update the status of a submission (e.g., pending, approved, rejected, on_hold)."""
    import json as _json
    from starlette.requests import Request
    from fastapi import Request as FRequest

    # Read raw body since we just need a simple field
    # (Using inline approach to avoid adding another Pydantic model)
    request = app.state  # We'll use the dependency injection below instead
    return {"error": "Use the proper endpoint"}


# Proper status update with body parsing
@app.put("/api/submissions/{submission_id}/status")
async def set_submission_status(submission_id: str, body: dict = None):
    """Set submission status. Body: {"status": "approved"}"""
    db = _load_db()
    record = db.get("submissions", {}).get(submission_id)
    if not record:
        raise HTTPException(status_code=404, detail="Submission not found")

    return {"error": "Use POST endpoint"}


class StatusUpdateRequest(BaseModel):
    status: str  # pending_review, approved, rejected, on_hold, completed
    note: str = ""


@app.post("/api/submissions/{submission_id}/status")
async def post_submission_status(submission_id: str, req: StatusUpdateRequest):
    """Update submission status and optional note."""
    valid_statuses = {
        "pending_review", "in_review", "approved", "rejected",
        "on_hold", "completed", "cancelled", "needs_revision",
    }
    if req.status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{req.status}'. Valid: {', '.join(sorted(valid_statuses))}"
        )

    db = _load_db()
    record = db.get("submissions", {}).get(submission_id)
    if not record:
        raise HTTPException(status_code=404, detail="Submission not found")

    old_status = record.get("status", "pending_review")
    record["status"] = req.status
    if req.note:
        if "notes" not in record:
            record["notes"] = []
        record["notes"].append({
            "text": req.note,
            "timestamp": datetime.now().isoformat(),
            "status_change": req.status,
        })
    record["updated_at"] = datetime.now().isoformat()
    _add_audit_event(
        record, "status_change",
        f"Status changed from '{old_status.replace('_', ' ')}' to '{req.status.replace('_', ' ')}'."
        + (f" Note: {req.note}" if req.note else ""),
        actor="user",
        metadata={"old_status": old_status, "new_status": req.status},
    )
    _save_db(db)

    return {"submission_id": submission_id, "status": req.status}


class PurchaseDetailsUpdate(BaseModel):
    """Fields from the Purchase Details form (mirrors MS Forms fields)."""
    accounting_line: str = ""
    type_of_request: str = ""  # Hardware, Software/Licensing, Contract Labor, Other
    future_cost_hardware: str = ""  # yes / no
    future_cost_software: str = ""  # yes / no
    renewal_cost_year1: str = ""
    custodian_code_shipping_notes: str = ""
    # Allow overriding AI-extracted fields too
    requestor_name: str | None = None
    department: str | None = None


@app.put("/api/submissions/{submission_id}/details")
async def update_purchase_details(submission_id: str, req: PurchaseDetailsUpdate):
    """Save purchase detail fields (accounting line, type of request, etc.)."""
    db = _load_db()
    record = db.get("submissions", {}).get(submission_id)
    if not record:
        raise HTTPException(status_code=404, detail="Submission not found")

    ed = record.get("extracted_data", {})
    # Update the MS Forms fields
    ed["accounting_line"] = req.accounting_line
    ed["type_of_request"] = req.type_of_request
    ed["future_cost_hardware"] = req.future_cost_hardware
    ed["future_cost_software"] = req.future_cost_software
    ed["renewal_cost_year1"] = req.renewal_cost_year1
    ed["custodian_code_shipping_notes"] = req.custodian_code_shipping_notes
    # Overrides for AI-extracted fields (if user corrects them)
    if req.requestor_name is not None:
        ed["requestor_name"] = req.requestor_name
    if req.department is not None:
        ed["department"] = req.department

    record["extracted_data"] = ed
    record["updated_at"] = datetime.now().isoformat()
    _add_audit_event(record, "details_updated", "Purchase details form updated.", actor="user")
    _save_db(db)

    return {"submission_id": submission_id, "message": "Purchase details saved"}


@app.get("/api/submissions/export")
async def export_submissions_csv():
    """Export all submissions as CSV for download."""
    import csv
    import io
    from starlette.responses import StreamingResponse

    db = _load_db()
    submissions = list(db.get("submissions", {}).values())
    submissions.sort(key=lambda s: s.get("uploaded_at", ""), reverse=True)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Submission ID", "Date", "Vendor Name", "Amount",
        "Description", "Department", "Requestor", "Contract Type",
        "Procurement Method", "Threshold Category", "Status",
        "Critical Issues", "Warnings", "Filename", "Funding Type",
        "Accounting Line", "Type of Request", "Future Cost - Hardware",
        "Future Cost - Software", "Year 1 Renewal Cost",
        "Custodian Code / Shipping / Notes",
    ])

    for s in submissions:
        ed = s.get("extracted_data", {})
        comp = s.get("compliance", {})
        issues = comp.get("issues", [])
        critical_count = len([i for i in issues if i.get("severity") == "critical"])
        warning_count = len([i for i in issues if i.get("severity") == "warning"])

        writer.writerow([
            s.get("submission_id", ""),
            s.get("uploaded_at", "")[:10] if s.get("uploaded_at") else "",
            ed.get("vendor_name", ""),
            ed.get("total_amount", 0),
            ed.get("scope_of_services", ""),
            ed.get("department", ""),
            ed.get("requestor_name", ""),
            ed.get("contract_type", ""),
            comp.get("procurement_method", ""),
            comp.get("threshold_category", ""),
            s.get("status", "pending_review"),
            critical_count,
            warning_count,
            s.get("filename", ""),
            ed.get("funding_type", "standard"),
            ed.get("accounting_line", ""),
            ed.get("type_of_request", ""),
            ed.get("future_cost_hardware", ""),
            ed.get("future_cost_software", ""),
            ed.get("renewal_cost_year1", ""),
            ed.get("custodian_code_shipping_notes", ""),
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=ocfl_submissions_{date.today().isoformat()}.csv"},
    )


@app.post("/api/generate-form")
async def generate_form(req: GenerateFormRequest):
    """Generate a filled procurement PDF form."""
    db = _load_db()
    record = db.get("submissions", {}).get(req.submission_id)
    if not record:
        raise HTTPException(status_code=404, detail="Submission not found")

    extracted = ExtractedData(**record["extracted_data"])
    out_filename = f"{req.submission_id}_{req.form_type}.pdf"
    out_path = FILLED_DIR / out_filename

    generators = {
        "piggyback_checklist": _generate_piggyback_checklist_pdf,
        "expediting_form": _generate_expediting_form_pdf,
    }

    generator = generators.get(req.form_type)
    if not generator:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown form_type '{req.form_type}'. "
                f"Available: {', '.join(generators.keys())}"
            ),
        )

    try:
        generator(extracted, req.field_overrides, out_path)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"PDF generation failed: {exc}",
        )

    return FileResponse(
        path=str(out_path),
        filename=out_filename,
        media_type="application/pdf",
    )


# ---------------------------------------------------------------------------
# Required Forms endpoint
# ---------------------------------------------------------------------------


@app.post("/api/submissions/{submission_id}/required-forms")
async def get_required_forms(submission_id: str):
    """Get or re-compute required forms for a submission."""
    db = _load_db()
    record = db.get("submissions", {}).get(submission_id)
    if not record:
        raise HTTPException(status_code=404, detail="Submission not found")

    extracted = ExtractedData(**record["extracted_data"])
    forms = determine_required_forms(extracted)

    # Update stored data
    db["submissions"][submission_id]["required_forms"] = forms
    _save_db(db)

    return {"submission_id": submission_id, "required_forms": forms}


# ---------------------------------------------------------------------------
# Pricing Intelligence endpoint
# ---------------------------------------------------------------------------


@app.post("/api/pricing-intelligence/{submission_id}")
async def get_pricing_intelligence(submission_id: str):
    """Run or re-run pricing intelligence for a submission."""
    db = _load_db()
    record = db.get("submissions", {}).get(submission_id)
    if not record:
        raise HTTPException(status_code=404, detail="Submission not found")

    extracted = ExtractedData(**{
        k: v for k, v in record["extracted_data"].items()
        if k in ExtractedData.model_fields
    })

    pricing = await run_pricing_intelligence(extracted)

    # Update stored data
    db["submissions"][submission_id]["pricing_intelligence"] = pricing
    _save_db(db)

    return {"submission_id": submission_id, "pricing_intelligence": pricing}


# ---------------------------------------------------------------------------
# Contract Search endpoints
# ---------------------------------------------------------------------------


@app.post("/api/contracts/search")
async def contract_search(req: ContractSearchRequest):
    """Search across all contract sources."""
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Search query is required")
    results = await search_contracts(req.query.strip(), req.search_type)
    return results


@app.get("/api/contracts/sources")
async def contract_sources():
    """List available contract search sources and their status."""
    sources = [
        {
            "name": "USASpending.gov",
            "type": "api",
            "active": True,
            "description": "Federal contract awards database",
            "auth_required": False,
        },
        {
            "name": "SAM.gov",
            "type": "api",
            "active": bool(SAM_API_KEY),
            "description": "System for Award Management — vendor registration and exclusions",
            "auth_required": True,
            "auth_configured": bool(SAM_API_KEY),
        },
    ]
    for link in DIRECT_SEARCH_LINKS:
        sources.append({
            "name": link["source"],
            "type": "web_search",
            "active": True,
            "description": f"Search {link['source']} (opens external site)",
            "auth_required": False,
        })
    return {"sources": sources}


# ---------------------------------------------------------------------------
# New Feature Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/analytics")
async def get_analytics():
    """Get comprehensive spend analytics."""
    return compute_spend_analytics()


@app.get("/api/vendor-scorecard/{vendor_name}")
async def get_vendor_scorecard(vendor_name: str):
    """Get performance scorecard for a specific vendor."""
    from urllib.parse import unquote
    return build_vendor_scorecard(unquote(vendor_name))


@app.get("/api/vendor-scorecards")
async def get_all_vendor_scorecards():
    """Get scorecards for all vendors."""
    return {"scorecards": build_all_vendor_scorecards()}


@app.get("/api/vendor-autofill/{vendor_name}")
async def get_autofill(vendor_name: str):
    """Get auto-fill suggestions for a returning vendor."""
    from urllib.parse import unquote
    return get_vendor_autofill(unquote(vendor_name))


@app.get("/api/expiration-alerts")
async def get_expiration_alerts_endpoint():
    """Get all upcoming contract/quote expirations."""
    return {"alerts": get_expiration_alerts()}


@app.get("/api/duplicate-check/{submission_id}")
async def check_duplicates(submission_id: str):
    """Check for duplicate or related purchases."""
    db = _load_db()
    record = db.get("submissions", {}).get(submission_id)
    if not record:
        raise HTTPException(status_code=404, detail="Submission not found")
    extracted = ExtractedData(**{
        k: v for k, v in record["extracted_data"].items()
        if k in ExtractedData.model_fields
    })
    alerts = detect_duplicates(extracted, submission_id)
    return {"submission_id": submission_id, "alerts": alerts}


@app.get("/api/submission/{submission_id}/timeline")
async def get_submission_timeline(submission_id: str):
    """Get the full audit trail / timeline for a submission."""
    db = _load_db()
    record = db.get("submissions", {}).get(submission_id)
    if not record:
        raise HTTPException(status_code=404, detail="Submission not found")
    return {
        "submission_id": submission_id,
        "timeline": record.get("audit_trail", []),
        "status": record.get("status", "pending_review"),
    }


@app.get("/api/submission/{submission_id}/routing")
async def get_submission_routing(submission_id: str):
    """Get the approval routing for a submission."""
    db = _load_db()
    record = db.get("submissions", {}).get(submission_id)
    if not record:
        raise HTTPException(status_code=404, detail="Submission not found")
    extracted = ExtractedData(**{
        k: v for k, v in record["extracted_data"].items()
        if k in ExtractedData.model_fields
    })
    compliance = ComplianceResult(**record.get("compliance", {}))
    route = determine_approval_route(extracted, compliance)
    return {"submission_id": submission_id, "routing": route.model_dump()}


@app.post("/api/nlq")
async def natural_language_query(req: NLQueryRequest):
    """Answer a natural language question about procurement data."""
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is required")
    return run_natural_language_query(req.question.strip())


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "OCFL Procurement Portal API"}


# ---------------------------------------------------------------------------
# Static files & SPA fallback
# ---------------------------------------------------------------------------

STATIC_DIR = BASE_DIR / "static"


@app.get("/")
async def serve_index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>OCFL Procurement Portal</h1><p>Static files not found.</p>")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
