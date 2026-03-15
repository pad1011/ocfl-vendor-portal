# OCFL Vendor Quote Validation Portal

Local demo of the Orange County FL procurement pre-submission portal. Upload a vendor quote PDF, extract key fields, validate against procurement rules, prefill the Piggyback Requisition Checklist, and attach a Certificate of Insurance.

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python portal_api_demo.py
```

Then open http://localhost:5002

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/api/upload` | Upload a vendor quote PDF |
| POST | `/api/validate` | Re-run validation for a submission |
| GET | `/api/forms/piggyback/prefill` | Get prefilled checklist values |
| POST | `/api/forms/piggyback/save` | Save checklist and generate PDF |
| POST | `/api/attachments/coi` | Upload Certificate of Insurance |
| GET | `/api/download/piggyback/{id}` | Download generated checklist PDF |
| GET | `/api/admin/rules` | List validation rules |

## Project Files

| File | Purpose |
|------|---------|
| `portal_api_demo.py` | Main Flask API server |
| `fill_acroform_tool.py` | List and fill PDF AcroForm fields |
| `create_sample_quote.py` | Generate a sample vendor quote PDF for testing |
| `openapi.yaml` | OpenAPI 3.0 specification |
| `postman_collection.json` | Postman collection for API testing |
| `requirements.txt` | Python dependencies |

## Testing with Postman

1. Import `postman_collection.json` into Postman
2. Start the server locally on port 5002
3. Run the requests in order: Upload Quote, Validate, Piggyback, COI

## Generate Sample PDFs

```bash
python create_sample_quote.py
```
