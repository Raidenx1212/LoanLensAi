import os
import fitz  # PyMuPDF


def create_sample_sanction_letter(output_path: str = "sample_sanction_letter.pdf"):
    """
    Generates a realistic 1-page Indian Bank Loan Sanction Letter PDF
    containing tabular Key Fact Statement (KFS) terms.
    """
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 size

    text_content = """
    HDFC BANK LIMITED - RETAIL ASSETS DIVISION
    SANCTION LETTER & KEY FACT STATEMENT (KFS)
    
    Date: 12-Nov-2024
    Sanction Reference: HDFC/PL/2024/984721
    Customer Name: Mr. Rahul Sharma
    
    Dear Rahul,
    We are pleased to inform you that your Personal Loan application has been sanctioned on the terms below:
    
    ---------------------------------------------------------------------------------
    LOAN KEY FACT STATEMENT (KFS) DETAILS
    ---------------------------------------------------------------------------------
    1. Sanctioned Principal Amount  : INR 5,00,000.00
    2. Rate of Interest             : 11.50% per annum (Floating reducing balance)
    3. Loan Tenure                  : 36 Months
    4. Equated Monthly Installment  : INR 16,490.00 per month
    5. Upfront Processing Fee       : INR 7,500.00 (exclusive of applicable 18% GST)
    6. Documentation & Stamp Fee    : INR 1,200.00 (exclusive of 18% GST)
    7. Bundled Credit Shield Cover  : INR 14,800.00 (deducted from disbursal)
    8. Prepayment / Foreclosure Fee : 3.00% penalty on principal outstanding if foreclosed within 12 months.
    ---------------------------------------------------------------------------------
    
    Disbursement will be made to your designated bank account post deduction of upfront fees and statutory taxes.
    
    Authorized Signatory
    HDFC Bank Ltd.
    """

    # Insert text into PDF
    page.insert_text((50, 60), text_content, fontsize=10, fontname="helv")
    doc.save(output_path)
    doc.close()
    print(f"Sample sanction letter PDF generated at: {output_path}")


if __name__ == "__main__":
    create_sample_sanction_letter()
