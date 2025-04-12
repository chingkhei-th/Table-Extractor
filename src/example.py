#!/usr/bin/env python3
"""
Example script demonstrating how to use the table extraction tool
"""

from table_extractor import TableExtractor

def main():
    # Define configuration
    config = {
        'dpi': 300,
        'denoise': True,
        'morph_close': True,
        'crop_padding': 1,
        'lang': 'en',
        'use_gpu': False
    }

    # Create table extractor
    extractor = TableExtractor(config)

    # Process a PDF file
    pdf_path = "sample.pdf"  # Replace with your PDF file
    output_dir = "output"

    # Extract tables and perform OCR
    extractor.process_pdf(pdf_path, output_dir)

    print(f"Extraction completed. Results saved to '{output_dir}' directory.")
    print(f"Merged CSV file: '{output_dir}/merged_table.csv'")

if __name__ == "__main__":
    main()