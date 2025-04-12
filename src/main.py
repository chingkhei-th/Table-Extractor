import os
import argparse
from table_extractor import TableExtractor

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract tables from PDFs')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--output_dir', '-o', default='data/output', help='Output directory')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for PDF to image conversion')
    parser.add_argument('--denoise', action='store_true', help='Apply denoising to images')
    parser.add_argument('--no-denoise', dest='denoise', action='store_false')
    parser.add_argument('--morph-close', action='store_true', help='Apply morphological closing')
    parser.add_argument('--no-morph-close', dest='morph_close', action='store_false')
    parser.add_argument('--crop-padding', type=int, default=1, help='Padding around cropped tables')
    parser.add_argument('--lang', default='en', help='Language for OCR (default: en)')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU for OCR if available')

    # Set defaults
    parser.set_defaults(denoise=True, morph_close=True, use_gpu=False)

    # Parse arguments
    args = parser.parse_args()

    # Check if the PDF file exists
    if not os.path.isfile(args.pdf_path):
        print(f"Error: PDF file '{args.pdf_path}' not found")
        return

    # Create configuration from arguments
    config = {
        'dpi': args.dpi,
        'denoise': args.denoise,
        'morph_close': args.morph_close,
        'crop_padding': args.crop_padding,
        'lang': args.lang,
        'use_gpu': args.use_gpu
    }

    # Create the TableExtractor
    extractor = TableExtractor(config=config)

    # Process the PDF
    extractor.process_pdf(args.pdf_path, args.output_dir)

    print(f"Table extraction completed. Results saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()