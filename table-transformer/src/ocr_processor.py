# ocr_processor.py
import numpy as np
from paddleocr import PaddleOCR
from tqdm import tqdm

# Initialize PaddleOCR - will be loaded once when module is imported
ocr = PaddleOCR(use_angle_cls=True, lang="en")


def extract_text_from_cells(table_image, cell_coordinates):
    """
    Extract text from table cells using OCR

    Args:
        table_image (PIL.Image): Table image
        cell_coordinates (list): List of cell coordinates by row

    Returns:
        dict: Dictionary of extracted text by row and column
    """
    # OCR row by row
    data = dict()
    max_num_columns = 0

    for idx, row in enumerate(tqdm(cell_coordinates, desc="Processing rows")):
        row_text = []

        for cell in row["cells"]:
            # Crop cell out of image
            cell_image = np.array(table_image.crop(cell["cell"]))

            # Apply OCR
            result = ocr.ocr(cell_image, cls=True)

            # Extract text from results
            if result and len(result) > 0 and result[0]:
                # PaddleOCR format: [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], [text, confidence]]
                text_parts = []
                for line in result:
                    if line:
                        for item in line:
                            if len(item) >= 2:  # Make sure we have text and confidence
                                text_parts.append(
                                    item[1][0]
                                )  # item[1][0] contains the text
                text = " ".join(text_parts)
                row_text.append(text)
            else:
                row_text.append("")  # Empty string for cells with no detected text

        if len(row_text) > max_num_columns:
            max_num_columns = len(row_text)

        data[idx] = row_text

    # Pad rows to ensure all have the same number of columns
    for row, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
            row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
        data[row] = row_data

    return data
