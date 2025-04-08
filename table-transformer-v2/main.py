from src.pdf_processor import PDFProcessor

def main():
    pdf_path = "../data/test-input/Test_statement.pdf"
    processor = PDFProcessor()
    merged_csv = processor.process_pdf(pdf_path)
    print(f"Merged CSV saved at: {merged_csv}")

if __name__ == "__main__":
    main()