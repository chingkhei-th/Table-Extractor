# pdf_to_csv.py (Main script)
import argparse
import os
from src.pdf_processor import convert_pdf_to_images
from src.table_detector import detect_tables
from src.structure_recognizer import recognize_structure
from src.ocr_processor import extract_text_from_cells
from src.utils import save_to_csv, merge_csv_files


def main():
    parser = argparse.ArgumentParser(
        description="Extract tables from PDF and save as CSV"
    )
    parser.add_argument(
        "--pdf_path",
        type=str,
        required=False,
        help="Path to the PDF file",
        default="../data/test-input/Test_statement.pdf",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data/output/table-transformer",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--detection_threshold",
        type=float,
        default=0.5,
        help="Threshold for table detection",
    )
    parser.add_argument(
        "--crop_padding", type=int, default=0, help="Padding for table crops"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Save visualization images"
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Convert PDF to images
    print("Converting PDF to images...")
    images = convert_pdf_to_images(args.pdf_path)

    all_page_data = []

    for page_num, image in enumerate(images):
        print(f"\nProcessing page {page_num+1}/{len(images)}")
        page_dir = os.path.join(args.output_dir, f"page_{page_num+1}")
        os.makedirs(page_dir, exist_ok=True)

        # Save the page image
        page_image_path = os.path.join(page_dir, "page.jpg")
        image.save(page_image_path)

        # Detect tables in the page
        tables = detect_tables(
            image, args.detection_threshold, args.visualize, page_dir
        )

        page_tables_data = []
        for table_idx, table_crop in enumerate(tables):
            print(f"  Processing table {table_idx+1}/{len(tables)}")

            # Recognize table structure
            cells, structure = recognize_structure(
                table_crop, args.visualize, page_dir, table_idx
            )

            # Extract text using OCR
            table_data = extract_text_from_cells(table_crop, structure)

            # Save individual table CSV
            table_csv_path = os.path.join(page_dir, f"table_{table_idx+1}.csv")
            save_to_csv(table_data, table_csv_path)

            page_tables_data.append(table_data)

        # Add page data to all data
        if page_tables_data:
            all_page_data.extend(page_tables_data)

    # Merge all CSV files
    if all_page_data:
        merged_csv_path = os.path.join(args.output_dir, "all_tables.csv")
        merge_csv_files(all_page_data, merged_csv_path)
        print(f"\nAll tables merged and saved to {merged_csv_path}")


if __name__ == "__main__":
    main()
