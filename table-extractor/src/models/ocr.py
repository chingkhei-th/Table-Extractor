"""OCR model implementation."""

import numpy as np
from paddleocr import PaddleOCR


class OCRModel:
    """OCR model for text extraction from images."""

    def __init__(self, config):
        """Initialize the OCR model.

        Args:
            config (TableExtractorConfig): Configuration object.
        """
        self.config = config

        # Initialize OCR
        print("Initializing PaddleOCR...")
        self.ocr = PaddleOCR(use_angle_cls=config.use_angle_cls, lang=config.ocr_lang)

    def apply_ocr(self, cropped_table, cell_coordinates):
        """Apply OCR to each cell in the table.

        Args:
            cropped_table (PIL.Image): Cropped table image.
            cell_coordinates (list): List of cell coordinates.

        Returns:
            tuple: (data dictionary, number of columns)
        """
        data = dict()
        max_num_columns = 0

        for idx, row in enumerate(cell_coordinates):
            row_text = []
            for cell in row["cells"]:
                # Crop cell out of image
                cell_image = np.array(cropped_table.crop(cell["cell"]))

                # Apply OCR
                result = self.ocr.ocr(cell_image, cls=True)

                if result and len(result) > 0 and result[0]:
                    # Extract text from results
                    text_parts = []
                    for line in result:
                        if line:
                            for item in line:
                                if (
                                    len(item) >= 2
                                ):  # Make sure we have the text and confidence
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
                row_data = row_data + [
                    "" for _ in range(max_num_columns - len(row_data))
                ]
            data[row] = row_data

        return data, max_num_columns
