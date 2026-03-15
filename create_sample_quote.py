# create_sample_quote.py
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import datetime

def make_sample_quote(path="sample_quote.pdf"):
    c = canvas.Canvas(path, pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, 750, "Vendor Quote")
    c.setFont("Helvetica", 11)
    c.drawString(40, 730, "Vendor Name: ABC Supplies, Inc.")
    c.drawString(40, 712, "Vendor Number: VEND-12345")
    c.drawString(40, 694, "Requisition Number: REQ-2026-009")
    c.drawString(40, 676, "Contract Number: ACS-2024-001")
    c.drawString(40, 658, f"Date Submitted: {datetime.date.today().strftime('%m/%d/%Y')}")
    c.drawString(40, 640, "Requestor Name: John D. Example")
    c.drawString(40, 622, "Requestor Phone: 407-836-1018")
    c.drawString(40, 600, "Scope of Services: Provide routine maintenance and materials for the X-100 unit.")
    c.drawString(40, 580, "Price quotes must be firm for 60 days.")

    # line items
    y = 540
    c.drawString(40, y, "Qty  Description                    Unit Price    Extended")
    y -= 18
    c.drawString(40, y, "10   Labor man hour                 $50.00         $500.00")
    y -= 16
    c.drawString(40, y, "1    Materials lump sum             $1,500.00      $1,500.00")
    y -= 16
    c.drawString(40, y, "TOTAL                                              $2,000.00")
    y -= 36

    c.save()
    print("Wrote", path)

if __name__ == "__main__":
    make_sample_quote()
