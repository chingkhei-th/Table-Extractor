from paddleocr import PaddleOCR
from tqdm import tqdm
from PIL import Image
from src.config import Config
import numpy as np


class OCRProcessor:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang=Config.OCR_LANG)

    def process_cells(self, cell_coordinates, cropped_table):
        """Perform OCR without rotation correction"""
        data = {}
        max_cols = 0

        for idx, row in enumerate(tqdm(cell_coordinates)):
            row_text = []
            for cell in row["cells"]:
                cell_img = np.array(cropped_table.crop(cell["cell"]))
                result = self.ocr.ocr(cell_img, cls=True)

                if result and len(result) > 0 and result[0]:
                    text = " ".join(
                        [res[1][0] for line in result if line for res in line]
                    )
                else:
                    text = ""
                row_text.append(text)

            max_cols = max(max_cols, len(row_text))
            data[idx] = row_text

        # Pad rows with empty strings
        for key in data:
            data[key] += [""] * (max_cols - len(data[key]))

        return data
