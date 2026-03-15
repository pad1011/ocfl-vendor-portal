#!/usr/bin/env python3
"""
Local demo server for the OCFL procurement portal.

What it does:
- Upload a quote PDF
- Extract a few core fields with pdfplumber + regex
- Return validation issues
- Prefill and save a Piggyback Requisition Checklist PDF
- Upload a COI

This is intentionally a local demo. It stores data under ./demo_data and keeps state in JSON.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import pdfplumber
from flask import Flask, jsonify, request, send_file
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics, ttfonts
from reportlab.pdfgen import canvas

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "demo_data"
SUBMISSIONS_DIR = DATA_DIR / "submissions"
DB_PATH = DATA_DIR / "db.json"
FONT_PATH = APP_DIR / "DejaVuSans.ttf"

DATA_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

FONT_EMBEDDED = False
if FONT_PATH.exists():
    pdfmetrics.registerFont(ttfonts.TTFont("DejaVuSans", str(FONT_PATH)))
    FONT_EMBEDDED = True


RULES = [
    {
        "id": "vendor_name_present",
        "severity": "critical",
        "description": "Vendor name must be present on the submission package."
    },
    {
        "id": "line_by_line_pricing",
        "severity": "critical",
        "description": "Line-by-line pricing evidence should be present."
    },
    {
        "id": "piggyback_checklist_completed",
        "severity": "warning",
        "description": "Piggyback Requisition Checklist should be completed for alternate contract source workflows."
    },
    {
        "id": "coi_uploaded",
        "severity": "warning",
        "description": "Certificate of Insurance should be attached before final submission."
    }
]


def load_db() -> Dict[str, Any]:
    if not DB_PATH.exists():
        return {"submissions": {}}
    return json.loads(DB_PATH.read_text(encoding="utf-8"))



def save_db(db: Dict[str, Any]) -> None:
    DB_PATH.write_text(json.dumps(db, indent=2), encoding="utf-8")



def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    text_pages: List[Dict[str, Any]] = []
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text_pages.append({"page": page_num, "text": text})
    return text_pages



def parse_basic_fields(text_pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    combined = "\n".join(p["text"] for p in text_pages)

    def find(regex: str):
        match = re.search(regex, combined, re.IGNORECASE)
        return match.group(1).strip() if match else None

    vendor_name = find(r"Vendor Name[:\s]*([A-Z0-9\-\., &'()]{3,200})")
    requisition_number = find(r"Requisition Number[:\s]*([A-Z0-9\-/]{3,50})")
    contract_number = find(r"Contract Number[:\s]*([A-Z0-9\-_\/]{3,80})")
    date_submitted = find(r"(?:Date Submitted|Submitted)[:\s]*([0-9]{1,2}[/\-.][0-9]{1,2}[/\-.][0-9]{2,4})")
    requestor_name = find(r"Requestor Name[:\s]*([A-Za-z][A-Za-z ,.'-]{2,120})")
    requestor_phone = find(r"(?:Requestor Phone|Phone)[:\s]*([0-9\(\)\-\s]{7,20})")

    line_items = []
    for page in text_pages:
        for line in page["text"].splitlines():
            if re.search(r"\$\s*[0-9,]+(?:\.[0-9]{2})?", line):
                line_items.append({"raw": line[:320], "page": page["page"]})

    manifest: Dict[str, Any] = {
        "vendor_name": vendor_name,
        "requisition_number": requisition_number,
        "contract_number": contract_number,
        "date_submitted": date_submitted,
        "requestor_name": requestor_name,
        "requestor_phone": requestor_phone,
        "department": None,
        "line_items": line_items,
        "confidence": {}
    }
    for key, value in manifest.items():
        if key == "confidence":
            continue
        if key == "line_items":
            manifest["confidence"][key] = "high" if value else "low"
        else:
            manifest["confidence"][key] = "high" if value else "low"
    return manifest



def build_evidence_from_manifest(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    evidence = {}
    if manifest.get("line_items"):
        evidence["Line-by-line pricing validation"] = {
            "source": "uploaded quote",
            "page": manifest["line_items"][0]["page"],
            "snippet": manifest["line_items"][0]["raw"]
        }
    if manifest.get("vendor_name"):
        evidence["Vendor Name"] = {
            "source": "uploaded quote",
            "page": 1,
            "snippet": f"Vendor Name: {manifest['vendor_name']}"
        }
    if manifest.get("contract_number"):
        evidence["Contract Number"] = {
            "source": "uploaded quote",
            "page": 1,
            "snippet": f"Contract Number: {manifest['contract_number']}"
        }
    return evidence



def validate_submission_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    manifest = record["manifest"]
    issues: List[Dict[str, Any]] = []
    evidence = build_evidence_from_manifest(manifest)

    if not manifest.get("vendor_name"):
        issues.append({
            "rule_id": "vendor_name_present",
            "severity": "critical",
            "message": "Vendor name is missing from the uploaded quote.",
            "evidence": []
        })

    if not manifest.get("line_items"):
        issues.append({
            "rule_id": "line_by_line_pricing",
            "severity": "critical",
            "message": "Missing or unreadable line-by-line pricing evidence.",
            "evidence": []
        })
    else:
        issues.append({
            "rule_id": "line_by_line_pricing",
            "severity": "info",
            "message": "Line-by-line pricing evidence detected.",
            "evidence": [evidence["Line-by-line pricing validation"]]
        })

    if manifest.get("contract_number") and not record.get("piggyback_saved"):
        issues.append({
            "rule_id": "piggyback_checklist_completed",
            "severity": "warning",
            "message": "Piggyback Requisition Checklist has not been completed yet.",
            "evidence": [evidence.get("Contract Number", {})]
        })

    if not record.get("coi_uploaded"):
        issues.append({
            "rule_id": "coi_uploaded",
            "severity": "warning",
            "message": "Certificate of Insurance has not been uploaded yet.",
            "evidence": []
        })

    return issues



def create_piggyback_pdf(submission_id: str, fields: Dict[str, Any], manifest: Dict[str, Any]) -> Path:
    out_path = SUBMISSIONS_DIR / submission_id / f"piggyback_{submission_id}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    font_name = "DejaVuSans" if FONT_EMBEDDED else "Helvetica"
    c = canvas.Canvas(str(out_path), pagesize=letter)
    c.setFont(font_name, 14)
    c.drawString(40, 750, "Piggyback Requisition Checklist")
    c.setFont(font_name, 11)
    c.drawString(40, 732, f"Submission ID: {submission_id}")

    y = 700
    def row(label: str, value: Any) -> None:
        nonlocal y
        c.drawString(40, y, label)
        c.drawString(240, y, str(value or ""))
        y -= 18

    row("Vendor Name:", fields.get("Vendor Name") or manifest.get("vendor_name"))
    row("Requisition Number:", fields.get("Requisition Number") or manifest.get("requisition_number"))
    row("Date Submitted:", fields.get("Date Submitted") or manifest.get("date_submitted"))
    row("Department/Division:", fields.get("Department/Division") or manifest.get("department"))
    row("Requestor Name:", fields.get("Requestor Name") or manifest.get("requestor_name"))
    row("Requestor Phone:", fields.get("Requestor Phone") or manifest.get("requestor_phone"))
    row("Contract Number:", fields.get("Contract Number") or manifest.get("contract_number"))

    y -= 10
    c.drawString(40, y, "Checklist")
    y -= 18
    checks = [
        ("Confirm Contract is on the ACS log", fields.get("Confirm Contract is on the ACS log", False)),
        ("Contract is not expired", fields.get("Contract is not expired", False)),
        ("Vendor name on contract matches quote", fields.get("Vendor name on contract matches quote", False)),
        ("Line-by-line pricing validation", fields.get("Line-by-line pricing validation", bool(manifest.get("line_items"))))
    ]
    for label, state in checks:
        mark = "X" if bool(state) else " "
        c.drawString(48, y, f"[{mark}] {label}")
        y -= 18

    y -= 10
    c.drawString(40, y, "Evidence snippets")
    y -= 16
    for item in manifest.get("line_items", [])[:6]:
        snippet = item["raw"]
        if len(snippet) > 90:
            snippet = snippet[:87] + "..."
        c.drawString(48, y, f"p{item['page']}: {snippet}")
        y -= 14
        if y < 60:
            c.showPage()
            c.setFont(font_name, 11)
            y = 740

    y -= 20
    c.drawString(40, y, "Prepared by: ____________________")
    c.drawString(300, y, "Date: ____________________")
    c.save()
    return out_path


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok", "font_embedded": FONT_EMBEDDED})


@app.get("/")
def root() -> Any:
    return (
        "<h2>OCFL Procurement Portal Demo</h2>"
        "<p>Use Postman with the included collection or POST a PDF to /api/upload.</p>"
        "<ul>"
        "<li>GET /health</li>"
        "<li>POST /api/upload</li>"
        "<li>POST /api/validate</li>"
        "<li>GET /api/forms/piggyback/prefill?submission_id=...</li>"
        "<li>POST /api/forms/piggyback/save</li>"
        "<li>POST /api/attachments/coi</li>"
        "</ul>"
    )


@app.post("/api/upload")
def upload() -> Any:
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    submission_id = str(uuid.uuid4())
    sub_dir = SUBMISSIONS_DIR / submission_id
    sub_dir.mkdir(parents=True, exist_ok=True)

    raw_bytes = file.read()
    upload_path = sub_dir / file.filename
    upload_path.write_bytes(raw_bytes)

    pages = extract_text_from_pdf_bytes(raw_bytes)
    manifest = parse_basic_fields(pages)
    manifest["department"] = request.form.get("department") or manifest.get("department")
    manifest["requestor_name"] = request.form.get("requestor_name") or manifest.get("requestor_name")
    manifest["requestor_phone"] = request.form.get("requestor_phone") or manifest.get("requestor_phone")
    manifest["requisition_number"] = request.form.get("requisition_number") or manifest.get("requisition_number")

    record = {
        "submission_id": submission_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "upload_filename": file.filename,
        "upload_path": str(upload_path),
        "manifest": manifest,
        "piggyback_saved": False,
        "piggyback_fields": {},
        "piggyback_pdf": None,
        "coi_uploaded": False,
        "coi_filename": None,
        "audit_log": [
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "action": "upload",
                "details": {"filename": file.filename}
            }
        ]
    }
    record["validation"] = validate_submission_record(record)

    db = load_db()
    db["submissions"][submission_id] = record
    save_db(db)

    return jsonify(
        {
            "submission_id": submission_id,
            "manifest": manifest,
            "validation": record["validation"],
            "links": {
                "prefill_piggyback": f"/api/forms/piggyback/prefill?submission_id={submission_id}",
                "validate": "/api/validate"
            }
        }
    ), 201


@app.post("/api/validate")
def validate() -> Any:
    payload = request.get_json(silent=True) or {}
    submission_id = payload.get("submission_id")
    if not submission_id:
        return jsonify({"error": "submission_id is required"}), 400

    db = load_db()
    record = db["submissions"].get(submission_id)
    if not record:
        return jsonify({"error": "Submission not found"}), 404

    record["validation"] = validate_submission_record(record)
    db["submissions"][submission_id] = record
    save_db(db)

    status = "fail" if any(item["severity"] == "critical" for item in record["validation"]) else "pass"
    return jsonify({"submission_id": submission_id, "status": status, "issues": record["validation"]})


@app.get("/api/forms/piggyback/prefill")
def piggyback_prefill() -> Any:
    submission_id = request.args.get("submission_id")
    if not submission_id:
        return jsonify({"error": "submission_id is required"}), 400

    db = load_db()
    record = db["submissions"].get(submission_id)
    if not record:
        return jsonify({"error": "Submission not found"}), 404

    manifest = record["manifest"]
    fields = {
        "Vendor Name": manifest.get("vendor_name"),
        "Requisition Number": manifest.get("requisition_number"),
        "Date Submitted": manifest.get("date_submitted") or datetime.utcnow().strftime("%m/%d/%Y"),
        "Department/Division": manifest.get("department"),
        "Requestor Name": manifest.get("requestor_name"),
        "Requestor Phone": manifest.get("requestor_phone"),
        "Contract Number": manifest.get("contract_number"),
        "Confirm Contract is on the ACS log": False,
        "Contract is not expired": False,
        "Vendor name on contract matches quote": False,
        "Line-by-line pricing validation": bool(manifest.get("line_items"))
    }
    evidence = build_evidence_from_manifest(manifest)
    confidence = {
        "Vendor Name": manifest["confidence"].get("vendor_name", "low"),
        "Requisition Number": manifest["confidence"].get("requisition_number", "low"),
        "Date Submitted": manifest["confidence"].get("date_submitted", "low"),
        "Department/Division": manifest["confidence"].get("department", "low"),
        "Requestor Name": manifest["confidence"].get("requestor_name", "low"),
        "Requestor Phone": manifest["confidence"].get("requestor_phone", "low"),
        "Contract Number": manifest["confidence"].get("contract_number", "low"),
        "Line-by-line pricing validation": manifest["confidence"].get("line_items", "low")
    }
    return jsonify({
        "submission_id": submission_id,
        "form": "Piggyback Requisition Checklist",
        "fields": fields,
        "confidence": confidence,
        "evidence": evidence
    })


@app.post("/api/forms/piggyback/save")
def piggyback_save() -> Any:
    payload = request.get_json(silent=True) or {}
    submission_id = payload.get("submission_id")
    fields = payload.get("fields")
    if not submission_id or not isinstance(fields, dict):
        return jsonify({"error": "submission_id and fields are required"}), 400

    db = load_db()
    record = db["submissions"].get(submission_id)
    if not record:
        return jsonify({"error": "Submission not found"}), 404

    manifest = record["manifest"]
    pdf_path = create_piggyback_pdf(submission_id, fields, manifest)
    record["piggyback_saved"] = True
    record["piggyback_fields"] = fields
    record["piggyback_pdf"] = str(pdf_path)
    record["audit_log"].append({
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "action": "piggyback_saved",
        "details": {"path": str(pdf_path)}
    })
    record["validation"] = validate_submission_record(record)

    db["submissions"][submission_id] = record
    save_db(db)

    return jsonify({
        "submission_id": submission_id,
        "saved": True,
        "pdf_url": f"/api/download/piggyback/{submission_id}",
        "validation": record["validation"]
    })


@app.post("/api/attachments/coi")
def upload_coi() -> Any:
    submission_id = request.form.get("submission_id")
    file = request.files.get("file")
    if not submission_id or not file:
        return jsonify({"error": "submission_id and file are required"}), 400

    db = load_db()
    record = db["submissions"].get(submission_id)
    if not record:
        return jsonify({"error": "Submission not found"}), 404

    sub_dir = SUBMISSIONS_DIR / submission_id
    coi_path = sub_dir / file.filename
    coi_path.write_bytes(file.read())

    record["coi_uploaded"] = True
    record["coi_filename"] = file.filename
    record["audit_log"].append({
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "action": "coi_uploaded",
        "details": {"filename": file.filename}
    })
    record["validation"] = validate_submission_record(record)
    db["submissions"][submission_id] = record
    save_db(db)

    return jsonify({"submission_id": submission_id, "coi_uploaded": True, "filename": file.filename})


@app.get("/api/download/piggyback/<submission_id>")
def download_piggyback(submission_id: str) -> Any:
    db = load_db()
    record = db["submissions"].get(submission_id)
    if not record or not record.get("piggyback_pdf"):
        return jsonify({"error": "Generated PDF not found"}), 404
    path = Path(record["piggyback_pdf"])
    return send_file(path, mimetype="application/pdf", as_attachment=True, download_name=path.name)


@app.get("/api/admin/rules")
def list_rules() -> Any:
    return jsonify({"rules": RULES})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5002"))
    app.run(host="0.0.0.0", port=port, debug=True)
