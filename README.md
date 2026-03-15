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

- `GET /health` – health check
- - `POST /api/upload` – upload a vendor quote PDF
  - - `POST /api/validate` – re-run validation for a submission
    - - `GET /api/forms/piggyback/prefill?submission_id=...` – get prefilled checklist values
      - - `POST /api/forms/piggyback/save` – save checklist and generate PDF
        - - `POST /api/attachments/coi` – upload Certificate of Insurance
          - - `GET /api/download/piggyback/<submission_id>` – download generated checklist PDF
            - - `GET /api/admin/rules` – list validation rules
             
              - ## Project Files
             
              - | File | Purpose |
              - |------|---------|
              - | `portal_api_demo.py` | Main Flask API server |
              - | `fill_acroform_tool.py` | List and fill PDF AcroForm fields |
              - | `create_sample_quote.py` | Generate a sample vendor quote PDF for testing |
              - | `openapi.yaml` | OpenAPI 3.0 specification |
              - | `postman_collection.json` | Postman collection for API testing |
              - | `requirements.txt` | Python dependencies |
             
              - ## Testing with Postman
             
              - 1. Import `postman_collection.json` into Postman
                2. 2. Start the server locally on port 5002
                   3. 3. Run the requests in order (Upload Quote → Validate → Piggyback → COI)
                     
                      4. ## Generate Sample PDFs
                     
                      5. ```bash
                         python create_sample_quote.py
                         ```
