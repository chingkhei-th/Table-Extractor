class Config:
    DETECTION_MODEL_NAME = "microsoft/table-transformer-detection"
    STRUCTURE_MODEL_NAME = "microsoft/table-structure-recognition-v1.1-all"
    DETECTION_THRESHOLDS = {"table": 0.5, "table rotated": 0.5, "no object": 10}
    OUTPUT_DIR = "../data/output/table-transformer-v2"
    DPI = 300
    CROP_PADDING = 10
    OCR_LANG = 'en'