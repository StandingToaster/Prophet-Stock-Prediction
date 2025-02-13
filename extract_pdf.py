from pdfminer.high_level import extract_text

def extract_text_from_pdf(pdf_path):
    """Extracts raw text from a PDF book"""
    text = extract_text(pdf_path)
    return text

# Extract text from The Intelligent Investor
pdf_text = extract_text_from_pdf("the-intelligent-investor.pdf")

# Save extracted text to a file for later use
with open("finance_knowledge.txt", "w", encoding="utf-8") as file:
    file.write(pdf_text)

print(" Extraction complete! Text saved in 'finance_knowledge.txt'")
