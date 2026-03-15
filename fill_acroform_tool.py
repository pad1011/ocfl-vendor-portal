#!/usr/bin/env python3
"""
List and fill PDF AcroForm fields.

Use cases:
  1) Inspect the official OCFL Piggyback Requisition Checklist PDF and dump field names.
  2) Fill the PDF with extracted values from the portal manifest.

Examples:
  python fill_acroform_tool.py list \
    --template Piggyback_Template.pdf \
    --out piggyback_fields.json

  python fill_acroform_tool.py skeleton \
    --template Piggyback_Template.pdf \
    --out sample_values.json

  python fill_acroform_tool.py fill \
    --template Piggyback_Template.pdf \
    --data sample_values.json \
    --out Piggyback_Filled.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pdfrw import PdfDict, PdfName, PdfObject, PdfReader, PdfString, PdfWriter

ANNOT_KEY = "/Annots"
ANNOT_FIELD_KEY = "/T"
ANNOT_PARENT_KEY = "/Parent"
ANNOT_SUBTYPE_KEY = "/Subtype"
ANNOT_WIDGET = "/Widget"
ANNOT_FIELD_TYPE = "/FT"
ANNOT_RECT_KEY = "/Rect"
ANNOT_AP_KEY = "/AP"
ANNOT_VAL_KEY = "/V"
ANNOT_AS_KEY = "/AS"


def _decode_pdf_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "to_unicode"):
        try:
            return value.to_unicode()
        except Exception:
            pass
    text = str(value)
    if text.startswith("(") and text.endswith(")"):
        return text[1:-1]
    return text


def _find_field_obj(widget: Any) -> Any:
    current = widget
    while current is not None:
        if current.get(ANNOT_FIELD_KEY):
            return current
        current = current.get(ANNOT_PARENT_KEY)
    return widget


def _field_type(field_obj: Any) -> Optional[str]:
    current = field_obj
    while current is not None:
        field_type = current.get(ANNOT_FIELD_TYPE)
        if field_type:
            return str(field_type)
        current = current.get(ANNOT_PARENT_KEY)
    return None


def _checkbox_on_value(widget: Any) -> str:
    ap = widget.get(ANNOT_AP_KEY)
    if not ap:
        return "Yes"
    normal = ap.get("/N") if isinstance(ap, dict) else None
    if not normal:
        return "Yes"
    keys = [str(k) for k in normal.keys()]
    for key in keys:
        if key != "/Off":
            return key.lstrip("/")
    return "Yes"


def iter_widgets(pdf: Any) -> Iterable[Tuple[int, Any, Any, Optional[str], Optional[str]]]:
    for page_num, page in enumerate(pdf.pages, start=1):
        annots = page.get(ANNOT_KEY) or []
        for annot in annots:
            widget = annot
            subtype = widget.get(ANNOT_SUBTYPE_KEY)
            if str(subtype) != ANNOT_WIDGET:
                continue
            field_obj = _find_field_obj(widget)
            name = _decode_pdf_string(field_obj.get(ANNOT_FIELD_KEY))
            ftype = _field_type(field_obj)
            yield page_num, widget, field_obj, name, ftype


def list_fields(template_path: Path) -> List[Dict[str, Any]]:
    pdf = PdfReader(str(template_path))
    seen: set[Tuple[str, int]] = set()
    results: List[Dict[str, Any]] = []
    for page_num, widget, field_obj, name, ftype in iter_widgets(pdf):
        if not name:
            continue
        key = (name, page_num)
        if key in seen:
            continue
        seen.add(key)
        rect = widget.get(ANNOT_RECT_KEY)
        options: List[str] = []
        if ftype == "/Btn":
            ap = widget.get(ANNOT_AP_KEY)
            if ap and ap.get("/N"):
                options = [str(k).lstrip("/") for k in ap["/N"].keys()]
        results.append(
            {
                "page": page_num,
                "field_name": name,
                "field_type": ftype or "unknown",
                "rect": [float(x) for x in rect] if rect else None,
                "options": options,
            }
        )
    return sorted(results, key=lambda item: (item["page"], item["field_name"]))


def write_skeleton(template_path: Path, out_path: Path) -> None:
    fields = list_fields(template_path)
    skeleton = {
        "_notes": [
            "Replace the placeholder values below with real portal values.",
            "Field names must exactly match the AcroForm field_name values discovered by the list command.",
            "Checkbox values may be true/false, 'Yes'/'Off', or the export value shown in options.",
        ],
        "values": {field["field_name"]: "" for field in fields},
    }
    out_path.write_text(json.dumps(skeleton, indent=2), encoding="utf-8")



def _set_text_value(target: Any, value: Any) -> None:
    target.update(PdfDict(V=PdfString.encode(str(value))))



def _set_checkbox_value(widget: Any, field_obj: Any, value: Any) -> None:
    truthy = str(value).strip().lower() in {"1", "true", "yes", "y", "checked", "on"}
    on_name = _checkbox_on_value(widget)
    export_name = on_name if truthy else "Off"
    target = field_obj if field_obj is not None else widget
    target.update(PdfDict(V=PdfName(export_name)))
    widget.update(PdfDict(AS=PdfName(export_name)))



def fill_pdf(template_path: Path, data_path: Path, out_path: Path) -> None:
    pdf = PdfReader(str(template_path))
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    values = payload["values"] if isinstance(payload, dict) and "values" in payload else payload

    if not isinstance(values, dict):
        raise ValueError("Input JSON must be an object or contain a top-level 'values' object")

    matched = set()
    for _page_num, widget, field_obj, name, ftype in iter_widgets(pdf):
        if not name or name not in values:
            continue
        value = values[name]
        matched.add(name)
        target = field_obj if field_obj is not None else widget
        if ftype == "/Btn":
            _set_checkbox_value(widget, target, value)
        else:
            _set_text_value(target, value)

    if pdf.Root and pdf.Root.AcroForm:
        pdf.Root.AcroForm.update(PdfDict(NeedAppearances=PdfObject("true")))

    PdfWriter().write(str(out_path), pdf)

    missing = sorted(set(values.keys()) - matched)
    if missing:
        print("Warning: these JSON keys did not match any fields in the template:")
        for item in missing:
            print(f"  - {item}")



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List or fill PDF AcroForm fields")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List AcroForm fields")
    p_list.add_argument("--template", required=True, type=Path, help="Path to the input PDF template")
    p_list.add_argument("--out", required=False, type=Path, help="Optional JSON output path")

    p_skel = sub.add_parser("skeleton", help="Create a skeleton JSON file from discovered fields")
    p_skel.add_argument("--template", required=True, type=Path, help="Path to the input PDF template")
    p_skel.add_argument("--out", required=True, type=Path, help="JSON output path")

    p_fill = sub.add_parser("fill", help="Fill a PDF template from a JSON values file")
    p_fill.add_argument("--template", required=True, type=Path, help="Path to the input PDF template")
    p_fill.add_argument("--data", required=True, type=Path, help="JSON values path")
    p_fill.add_argument("--out", required=True, type=Path, help="Filled PDF output path")

    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list":
        fields = list_fields(args.template)
        text = json.dumps(fields, indent=2)
        if args.out:
            args.out.write_text(text, encoding="utf-8")
            print(f"Wrote {args.out}")
        else:
            print(text)
    elif args.command == "skeleton":
        write_skeleton(args.template, args.out)
        print(f"Wrote {args.out}")
    elif args.command == "fill":
        fill_pdf(args.template, args.data, args.out)
        print(f"Wrote {args.out}")
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
