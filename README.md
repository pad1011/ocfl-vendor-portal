# OCFL Procurement Portal

Purchase request submission portal for Orange County Government, Florida — Information Systems & Services (ISS) Department.

## Features

- **Document Upload & AI Extraction**: Upload vendor quotes/invoices; AI automatically extracts vendor name, amount, terms, and more
- **Compliance Engine**: Checks all uploads against the full OCFL Procurement Procedures Manual (Rev. 06-25), flags missing documents, threshold violations, and M/WBE requirements
- **Quick Reference**: Searchable threshold tables for both Standard (County-funded) and Federal procurement methods
- **Form Pre-fill**: Extracted data is mapped to the official OCFL purchase request form fields

## Procurement Thresholds (Standard / County-Funded)

| Amount | Method | Requirements |
|--------|--------|-------------|
| $0 – $10,000 | P-Card / Small Purchase | Single transaction or direct purchase |
| $10,000 – $150,000 | Informal Quotes (RFQ) | Min. 3 written quotes; 1 M/WBE vendor |
| $150,000+ | Formal Solicitation | IFB or RFP, public advertisement |
| $500,000+ | BCC Approval | Board of County Commissioners approval |

## Tech Stack

- **Backend**: Python / FastAPI
- **Frontend**: Single-page application (vanilla JS)
- **AI**: Claude Sonnet for document extraction
- **PDF**: pikepdf for AcroForm filling

## Running Locally

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000 in your browser.

## Project Structure

```
├── server.py              # FastAPI backend (API + compliance engine)
├── fill_acroform_tool.py  # PDF AcroForm filler utility
├── static/
│   └── index.html         # Frontend SPA
├── uploads/               # Uploaded documents (gitignored)
├── filled/                # Generated filled PDFs (gitignored)
└── requirements.txt
```
