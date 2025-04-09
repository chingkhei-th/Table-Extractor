"""Main module for table extraction."""

import os
import torch
from tqdm import tqdm

from src.config import TableExtractorConfig
from src.models import TableDetectionModel, TableStructureModel, OCRModel
from src.column_detection import TemplateColumnDetector, validate_cell_structure
from src.utils import (
    convert_pdf_to_images,
    get_transform,
    visualize_detected_tables,
    visualize_table_structure,
    save_csv,
    merge_csvs,
)


class TableExtractor:
    """Main class for extracting tables from PDF documents."""

    def __init__(self, config=None):
        """Initialize the table extractor with models and configurations.

        Args:
            config (TableExtractorConfig, optional): Configuration object.
                If None, default configuration is used.
        """
        # Set configuration
        self.config = config if config else TableExtractorConfig()

        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Initialize models
        self.detection_model = TableDetectionModel(self.config, self.device)
        self.structure_model = TableStructureModel(self.config, self.device)
        self.ocr_model = OCRModel(self.config)

        # Initialize column detector
        self.column_detector = TemplateColumnDetector(
            self.config,
            self.detection_model,
            self.structure_model,
            self.device
        )

        # Set up transforms
        self.detection_transform = get_transform(self.config.max_detection_size, is_detection=True)
        self.structure_transform = get_transform(self.config.max_structure_size, is_detection=False)

    def process_pdf(self):
        """Process a PDF file using first page columns as template for all pages.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            str: Path to the merged CSV file, or None if no tables found.
        """
        # Convert PDF to images
        images = convert_pdf_to_images(pdf_path=self.config.input_pdf_path, dpi=self.config.pdf_dpi)

        # First, process the first page to get column structure
        print(f"\nProcessing first page to extract column structure template")
        first_page = images[0]
        self.column_detector.template_columns = self.column_detector.extract_column_template(first_page)

        # Process each page with the template columns
        all_data = []

        for page_idx, image in enumerate(tqdm(images, desc="Processing pages")):
            page_num = page_idx + 1
            print(f"\nProcessing page {page_num}/{len(images)}")

            page_data = self.process_image(image, page_num)
            all_data.append(page_data)

        # Merge all tables into a single CSV
        if any(all_data):
            merged_csv = merge_csvs(all_data, self.config.output_dir, header_page=0)
            print(f"Processing complete. Results saved to {self.config.output_dir}")
            return merged_csv
        else:
            print("No tables were found in the PDF.")
            return None

    def process_image(self, image, page_num):
        """Process a single page image to extract tables.

        Args:
            image (PIL.Image): Page image.
            page_num (int): Page number.

        Returns:
            list: List of tuples (table data, column count) for the page.
        """
        # Save original image
        orig_image_path = os.path.join(self.config.output_dir, "original", f"page_{page_num}.jpg")
        image.save(orig_image_path)

        # Table detection
        pixel_values = self.detection_transform(image).unsqueeze(0).to(self.device)
        outputs = self.detection_model.predict(pixel_values)

        # Extract table objects
        objects = self.detection_model.outputs_to_objects(outputs, image.size)

        if not objects:
            print(f"No tables detected on page {page_num}")
            return None

        # Visualize detected tables
        detected_image = visualize_detected_tables(image, objects)
        detected_image = detected_image.convert("RGB")
        detected_image_path = os.path.join(self.config.output_dir, "detected", f"page_{page_num}_detected.jpg")
        detected_image.save(detected_image_path)

        # Process each detected table
        all_page_data = []
        max_columns = 0

        for table_idx, table_crop_info in enumerate(self.detection_model.objects_to_crops(image, objects)):
            # Get the cropped table image
            cropped_table = table_crop_info['image']
            cropped_table_path = os.path.join(self.config.output_dir, "cropped", f"page_{page_num}_table_{table_idx}.jpg")
            cropped_table.save(cropped_table_path)

            # Apply structure recognition
            pixel_values = self.structure_transform(cropped_table).unsqueeze(0).to(self.device)
            structure_outputs = self.structure_model.predict(pixel_values)

            # Extract cell structure
            cells = self.structure_model.outputs_to_objects(structure_outputs, cropped_table.size)

            # Apply template-based columns if available
            if self.column_detector.template_columns:
                cells = self.column_detector.apply_template_to_table(cropped_table, cells)

            # Visualize structure
            structure_image = visualize_table_structure(cropped_table, cells)
            structure_image_path = os.path.join(self.config.output_dir, "structure", f"page_{page_num}_table_{table_idx}_structure.jpg")
            structure_image.save(structure_image_path)

            # Get cell coordinates
            cell_coordinates = self.structure_model.get_cell_coordinates_by_row(cells)

            # Validate and fix cell structure
            cell_coordinates = validate_cell_structure(cell_coordinates)

            if not cell_coordinates:
                print(f"No rows/columns detected in table {table_idx} on page {page_num}")
                continue

            # Apply OCR
            table_data, table_columns = self.ocr_model.apply_ocr(cropped_table, cell_coordinates)

            # Update max columns if this table has more
            max_columns = max(max_columns, table_columns)

            # Save individual table CSV
            table_csv_path = os.path.join(self.config.output_dir, "csv", f"page_{page_num}_table_{table_idx}.csv")
            save_csv(table_data, table_csv_path)

            # Add to page data
            all_page_data.append((table_data, table_columns))

        return all_page_data
