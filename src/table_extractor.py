import os
import cv2
import numpy as np
import pdf2image
from typing import List, Tuple, Dict, Any, Optional
import sys

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
                   table_regions: Dict[int, List[Tuple[int, int, int, int]]]) -> Dict[int, List[np.ndarray]]:
        """
        Crop detected tables from images

        Args:
            images (List[np.ndarray]): List of page images
            table_regions (Dict[int, List[Tuple[int, int, int, int]]]): Dictionary mapping page numbers to table regions

        Returns:
            Dict[int, List[np.ndarray]]: Dictionary mapping page numbers to lists of cropped table images
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
                page_tables.append(cropped)

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

    def process_pdf(self, pdf_path: str, output_dir: str) -> None:
        """
        Process a PDF file and extract tables

        Args:
            pdf_path (str): Path to the PDF file
            output_dir (str): Directory to save the output
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Load PDF
        images = self.load_pdf(pdf_path)

        # Detect tables
        table_regions = self.detect_tables(images)

        # Crop tables
        cropped_tables = self.crop_tables(images, table_regions)

        # Process each table
        for page_num, tables in cropped_tables.items():
            for table_idx, table_img in enumerate(tables):
                # Detect rows and columns
                rows, cols = self.detect_grid(table_img)

                # Draw grid
                grid_img = self.draw_grid(table_img, rows, cols)

                # Save output
                output_path = os.path.join(output_dir, f"page_{page_num}_table_{table_idx}.png")
                cv2.imwrite(output_path, grid_img)
                print(f"Saved table: {output_path}")