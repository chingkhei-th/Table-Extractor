import os
import cv2
import numpy as np
import pdf2image
import pandas as pd
import csv
from typing import List, Tuple, Dict, Any, Optional
import sys
from paddleocr import PaddleOCR

# Add script directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'script'))

from detect_table import TableDetector

class TableExtractor:
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the TableExtractor

        Args:
            config (Dict[str, Any], optional): Configuration dictionary
        """
        self.config = config or {}
        self.detector = TableDetector(config=self.config)
        # Initialize PaddleOCR
        use_gpu = self.config.get('use_gpu', False)
        lang = self.config.get('lang', 'en')
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu)

    def load_pdf(self, pdf_path: str) -> List[np.ndarray]:
        """
        Load a PDF file and convert it to images

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            List[np.ndarray]: List of page images
        """
        print(f"Loading PDF: {pdf_path}")

        # Convert PDF to images using pdf2image
        dpi = self.config.get('dpi', 300)
        pages = pdf2image.convert_from_path(pdf_path, dpi=dpi)

        # Convert PIL images to OpenCV format
        images = []
        for page in pages:
            # Convert PIL Image to numpy array
            img = np.array(page)
            # Convert RGB to BGR (OpenCV format)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            images.append(img)

        print(f"Converted {len(images)} pages to images")
        return images

    def detect_tables(self, images: List[np.ndarray]) -> Dict[int, List[Tuple[int, int, int, int]]]:
        """
        Detect tables in images

        Args:
            images (List[np.ndarray]): List of page images

        Returns:
            Dict[int, List[Tuple[int, int, int, int]]]: Dictionary mapping page numbers to table regions
        """
        table_regions = {}

        for page_num, img in enumerate(images):
            # Detect tables on the current page
            regions = self.detector.detect_tables(img, page_num)

            if regions:
                table_regions[page_num] = regions
                print(f"Page {page_num}: {len(regions)} tables detected")
            else:
                print(f"Page {page_num}: No tables detected")

        return table_regions

    def crop_tables(self, images: List[np.ndarray],
                   table_regions: Dict[int, List[Tuple[int, int, int, int]]]) -> Dict[int, List[Tuple[np.ndarray, Tuple[int, int, int, int]]]]:
        """
        Crop detected tables from images

        Args:
            images (List[np.ndarray]): List of page images
            table_regions (Dict[int, List[Tuple[int, int, int, int]]]): Dictionary mapping page numbers to table regions

        Returns:
            Dict[int, List[Tuple[np.ndarray, Tuple[int, int, int, int]]]]: Dictionary mapping page numbers to lists of (cropped table images, regions)
        """
        cropped_tables = {}

        for page_num, regions in table_regions.items():
            if page_num >= len(images):
                continue

            img = images[page_num]
            page_tables = []

            for i, (x, y, w, h) in enumerate(regions):
                # Add a small padding around the table
                padding = self.config.get('crop_padding', 10)
                x_min = max(0, x - padding)
                y_min = max(0, y - padding)
                x_max = min(img.shape[1], x + w + padding)
                y_max = min(img.shape[0], y + h + padding)

                # Crop the table
                cropped = img[y_min:y_max, x_min:x_max]
                page_tables.append((cropped, (x_min, y_min, x_max-x_min, y_max-y_min)))

            cropped_tables[page_num] = page_tables

        return cropped_tables

    def detect_grid(self, table_img: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        Detect rows and columns in a table image

        Args:
            table_img (np.ndarray): Table image

        Returns:
            Tuple[List[int], List[int]]: Lists of row and column positions
        """
        # Convert to grayscale
        gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)

        # Apply binary thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Create kernel for horizontal and vertical line detection
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

        # Detect horizontal lines
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        h_contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Detect vertical lines
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        v_contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract row positions
        rows = []
        for contour in h_contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Add the middle of the line
            rows.append(y + h // 2)

        # Extract column positions
        cols = []
        for contour in v_contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Add the middle of the line
            cols.append(x + w // 2)

        # Sort rows and columns
        rows.sort()
        cols.sort()

        return rows, cols

    def draw_grid(self, table_img: np.ndarray, rows: List[int], cols: List[int]) -> np.ndarray:
        """
        Draw grid lines on a table image

        Args:
            table_img (np.ndarray): Table image
            rows (List[int]): Row positions
            cols (List[int]): Column positions

        Returns:
            np.ndarray: Image with grid lines drawn
        """
        # Create a copy of the image
        result = table_img.copy()

        # Draw horizontal lines (rows)
        for y in rows:
            cv2.line(result, (0, y), (result.shape[1], y), (0, 0, 255), 2)

        # Draw vertical lines (columns)
        for x in cols:
            cv2.line(result, (x, 0), (x, result.shape[0]), (0, 0, 255), 2)

        return result

    def extract_cell_text(self, table_img: np.ndarray, rows: List[int], cols: List[int]) -> List[List[str]]:
        """
        Extract text from table cells using OCR

        Args:
            table_img (np.ndarray): Table image
            rows (List[int]): Row positions
            cols (List[int]): Column positions

        Returns:
            List[List[str]]: 2D array of cell text
        """
        # Add boundary rows and columns
        all_rows = [0] + rows + [table_img.shape[0]]
        all_cols = [0] + cols + [table_img.shape[1]]

        # Initialize table data
        table_data = []

        # Process each row
        for i in range(len(all_rows) - 1):
            row_data = []
            y1 = all_rows[i]
            y2 = all_rows[i + 1]

            for j in range(len(all_cols) - 1):
                x1 = all_cols[j]
                x2 = all_cols[j + 1]

                # Extract cell image
                cell_img = table_img[y1:y2, x1:x2]

                # Skip very small cells (likely noise)
                if cell_img.shape[0] < 5 or cell_img.shape[1] < 5:
                    row_data.append("")
                    continue

                # Run OCR on the cell
                result = self.ocr.ocr(cell_img, cls=True)

                # Extract text (handle empty results)
                if result and result[0]:
                    # PaddleOCR returns a list for each text region
                    cell_text = " ".join([line[1][0] for line in result[0] if line[1][0].strip()])
                    row_data.append(cell_text)
                else:
                    row_data.append("")

            table_data.append(row_data)

        # Trim empty rows and columns
        table_data = self._trim_table_data(table_data)

        return table_data

    def _trim_table_data(self, table_data: List[List[str]]) -> List[List[str]]:
        """
        Remove empty rows and columns from the table data

        Args:
            table_data (List[List[str]]): Original table data with possible empty rows/columns

        Returns:
            List[List[str]]: Trimmed table data
        """
        if not table_data or not table_data[0]:
            return table_data

        # Ensure all rows have the same number of columns
        max_cols = max(len(row) for row in table_data)
        table_data = [row + [''] * (max_cols - len(row)) for row in table_data]

        # Find first and last non-empty rows
        first_row = 0
        last_row = len(table_data) - 1

        # Find first non-empty row
        for i, row in enumerate(table_data):
            if any(cell.strip() for cell in row):
                first_row = i
                break

        # Find last non-empty row
        for i in range(len(table_data) - 1, -1, -1):
            if any(cell.strip() for cell in table_data[i]):
                last_row = i
                break

        # If no content rows found, return a single empty row
        if first_row > last_row:
            return [[]]

        # Find first and last non-empty columns
        first_col = 0
        last_col = max_cols - 1

        # Find first non-empty column
        for j in range(max_cols):
            if any(table_data[i][j].strip() for i in range(first_row, last_row + 1)):
                first_col = j
                break

        # Find last non-empty column
        for j in range(max_cols - 1, -1, -1):
            if any(table_data[i][j].strip() for i in range(first_row, last_row + 1)):
                last_col = j
                break

        # Extract the trimmed table
        trimmed_data = []
        for i in range(first_row, last_row + 1):
            row = table_data[i][first_col:last_col + 1]
            trimmed_data.append(row)

        return trimmed_data

    def save_as_csv(self, table_data: List[List[str]], output_path: str) -> None:
        """
        Save table data as CSV

        Args:
            table_data (List[List[str]]): 2D array of cell text
            output_path (str): Path to save the CSV file
        """
        # Normalize table data to ensure all rows have the same number of columns
        max_cols = max(len(row) for row in table_data) if table_data else 0
        normalized_data = [row + [''] * (max_cols - len(row)) for row in table_data]

        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for row in normalized_data:
                writer.writerow(row)

        print(f"Saved CSV: {output_path}")

    def merge_csv_files(self, csv_dir: str, output_path: str) -> None:
        """
        Merge CSV files by table structure (number of columns)

        Args:
            csv_dir (str): Directory containing CSV files
            output_path (str): Base path to save the merged CSV files
        """
        # Get all CSV files in the directory
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

        # Sort by page number and table index
        def get_page_table(filename):
            # Extract page number and table index from filename format "page_X_table_Y.csv"
            parts = filename.split('_')
            page_num = int(parts[1])
            table_num = int(parts[3].split('.')[0])
            return (page_num, table_num)

        csv_files.sort(key=get_page_table)

        # Group files by table structure (number of columns)
        structure_groups = {}  # {column_count: [(filename, table_data), ...]}

        for csv_file in csv_files:
            file_path = os.path.join(csv_dir, csv_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                table_data = list(reader)

                # Skip empty tables
                if not table_data:
                    continue

                # Get number of columns from the first row
                # Some rows might have different lengths, so we use the max length of all rows
                col_count = max(len(row) for row in table_data) if table_data else 0

                if col_count not in structure_groups:
                    structure_groups[col_count] = []

                structure_groups[col_count].append((csv_file, table_data))

        # Generate merged files for each structure group
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        base_dir = os.path.dirname(output_path)

        for col_count, file_data_pairs in structure_groups.items():
            # Skip empty groups
            if not file_data_pairs:
                continue

            merged_data = []
            filenames = []

            for filename, table_data in file_data_pairs:
                # Add separator row between tables if needed
                if merged_data:
                    merged_data.append([])  # Add a single empty row as separator

                merged_data.extend(table_data)
                filenames.append(filename)

            # Create output filename with structure identifier
            if len(structure_groups) > 1:
                # If there are multiple structure groups, include column count in filename
                output_file = os.path.join(base_dir, f"{base_name}_{col_count}cols.csv")
            else:
                # If there's only one structure group, use the original filename
                output_file = output_path

            # Write merged data to CSV
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                for row in merged_data:
                    writer.writerow(row)

            print(f"Merged {len(file_data_pairs)} tables with {col_count} columns into: {output_file}")
            print(f"   Tables included: {', '.join(filenames)}")

        if not structure_groups:
            print("No CSV files found to merge.")

    def process_pdf(self, pdf_path: str, output_dir: str) -> None:
        """
        Process a PDF file and extract tables

        Args:
            pdf_path (str): Path to the PDF file
            output_dir (str): Directory to save the output
                Structure:
                - output/
                  ├── merged_table_Xcols.csv (Multiple files based on structure)
                  ├── cropped/
                  ├── csv/
                  ├── detected/
                  ├── original/
                  └── structure/
        """
        # Create output directory structure
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'cropped'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'csv'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'detected'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'original'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'structure'), exist_ok=True)

        # Load PDF
        images = self.load_pdf(pdf_path)

        # Save original images
        for i, img in enumerate(images):
            original_path = os.path.join(output_dir, 'original', f"page_{i}.png")
            cv2.imwrite(original_path, img)

        # Detect tables
        table_regions = self.detect_tables(images)

        # Process each page with tables
        cropped_tables = self.crop_tables(images, table_regions)

        # Process each table
        for page_num, tables in cropped_tables.items():
            page_img = images[page_num].copy()

            # Draw detected regions on the original image
            for i, (table_img, region) in enumerate(tables):
                x, y, w, h = region
                # Draw rectangle around table
                cv2.rectangle(page_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Save cropped table
                cropped_path = os.path.join(output_dir, 'cropped', f"page_{page_num}_table_{i}.png")
                cv2.imwrite(cropped_path, table_img)

                # Detect rows and columns
                rows, cols = self.detect_grid(table_img)

                # Draw grid lines on the table
                grid_img = self.draw_grid(table_img, rows, cols)

                # Save table with grid structure
                structure_path = os.path.join(output_dir, 'structure', f"page_{page_num}_table_{i}.png")
                cv2.imwrite(structure_path, grid_img)

                # Extract text from cells
                table_data = self.extract_cell_text(table_img, rows, cols)

                # Save as CSV
                csv_path = os.path.join(output_dir, 'csv', f"page_{page_num}_table_{i}.csv")
                self.save_as_csv(table_data, csv_path)

            # Save image with detected tables
            detected_path = os.path.join(output_dir, 'detected', f"page_{page_num}.png")
            cv2.imwrite(detected_path, page_img)

        # Merge all CSV files
        csv_dir = os.path.join(output_dir, 'csv')
        merged_csv_path = os.path.join(output_dir, 'merged_table.csv')
        self.merge_csv_files(csv_dir, merged_csv_path)