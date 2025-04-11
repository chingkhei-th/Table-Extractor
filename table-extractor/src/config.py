"""Configuration settings for the table extractor."""

import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class TableExtractorConfig:
    """Configuration class for the table extractor."""

    # I/O configuration
    input_pdf_path: str = "../data/test-input/Test_statement.pdf"
    output_dir: str = "../data/output/table-extractor"

    # Image processing configuration
    max_detection_size: int = 800
    max_structure_size: int = 1000
    crop_padding: int = 5
    pdf_dpi: int = 400  # DPI for PDF to image conversion

    # Model configuration
    detection_model_name: str = "microsoft/table-transformer-detection"
    detection_model_revision: str = "no_timm"
    structure_model_name: str = "microsoft/table-structure-recognition-v1.1-all"
    ocr_lang: str = "en"
    use_angle_cls: bool = True

    # Detection thresholds
    detection_class_thresholds: Dict[str, float] = None

    def __post_init__(self):
        """Initialize default values after initialization."""
        if self.detection_class_thresholds is None:
            self.detection_class_thresholds = {
                "table": 0.9,
                "table rotated": 0.5,
                "no object": 10,
            }

        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "original"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "detected"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "cropped"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "structure"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "csv"), exist_ok=True)
