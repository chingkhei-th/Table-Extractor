#!/usr/bin/env python
"""Command-line entry point for table extraction."""

import argparse
from src.config import TableExtractorConfig
from src.main import TableExtractor


def main():
    """Parse command-line arguments and run the table extractor."""
    parser = argparse.ArgumentParser(description="Extract tables from PDF documents")

    # Input and output options
    parser.add_argument("--pdf_path", type=str, help="Path to the PDF file (optional, uses default from config if not provided)")
    parser.add_argument("--output", type=str, default="../data/output/table-extractor", help="Output directory")

    # Image processing options
    parser.add_argument(
        "--dpi", type=int, default=200, help="DPI for PDF to image conversion"
    )
    parser.add_argument(
        "--max-detection-size",
        type=int,
        default=800,
        help="Maximum size for detection model input",
    )
    parser.add_argument(
        "--max-structure-size",
        type=int,
        default=1000,
        help="Maximum size for structure model input",
    )
    parser.add_argument(
        "--crop-padding", type=int, default=10, help="Padding around table crops"
    )

    # Model options
    parser.add_argument(
        "--detection-model",
        type=str,
        default="microsoft/table-transformer-detection",
        help="Detection model name or path",
    )
    parser.add_argument(
        "--structure-model",
        type=str,
        default="microsoft/table-structure-recognition-v1.1-all",
        help="Structure model name or path",
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=0.5,
        help="Detection confidence threshold",
    )
    parser.add_argument("--ocr-lang", type=str, default="en", help="OCR language")

    args = parser.parse_args()

    # Create configuration
    config = TableExtractorConfig(
        input_pdf_path=args.pdf_path if args.pdf_path else TableExtractorConfig().input_pdf_path,
        output_dir=args.output,
        pdf_dpi=args.dpi,
        max_detection_size=args.max_detection_size,
        max_structure_size=args.max_structure_size,
        crop_padding=args.crop_padding,
        detection_model_name=args.detection_model,
        structure_model_name=args.structure_model,
        ocr_lang=args.ocr_lang,
        detection_class_thresholds={
            "table": args.detection_threshold,
            "table rotated": args.detection_threshold,
            "no object": 10,
        },
    )

    # Initialize and run extractor
    extractor = TableExtractor(config)
    extractor.process_pdf()  # No argument needed, path is in config


if __name__ == "__main__":
    main()
