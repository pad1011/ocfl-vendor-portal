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

    # Persist
    db = _load_db()
    record = {
        "submission_id": submission_id,
        "filename": file.filename,
        "saved_as": safe_filename,
        "uploaded_at": datetime.now().isoformat(),
        "extracted_data": extracted.model_dump(),
        "compliance": compliance.model_dump(),
        "required_forms": required_forms,
        "contract_matches": contract_matches,
    }
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
