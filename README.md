# PDF Table Extraction Project

This project extracts tables from multi-page PDF files, recognizes their structure, and extracts text using OCR. It outputs both individual CSV files for each table and a merged CSV file containing all tables.

## Features

- Process multi-page PDF documents
- Detect tables in each page using Table Transformer
- Recognize table structure (rows and columns)
- Extract text from cells using PaddleOCR
- Generate CSV files for each table
- Merge all tables into a single CSV file
- Visualize detection and recognition results

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the main script with the path to your PDF file:

```bash
python pdf_to_csv.py --pdf_path "your_document.pdf" --output_dir "output" --visualize
```

### Arguments:

- `--pdf_path`: Path to the PDF file (required)
- `--output_dir`: Directory to save output files (default: "output")
- `--detection_threshold`: Threshold for table detection (default: 0.5)
- `--crop_padding`: Padding for table crops (default: 0)
- `--visualize`: Save visualization images (optional flag)

## Output

The script creates the following directory structure:

```
output/
├── all_tables.csv                  # Merged CSV with all tables
├── page_1/
│   ├── page.jpg                    # Original page image
│   ├── detected_tables.jpg         # Visualization of detected tables
│   ├── table_1.csv                 # CSV for first table
│   ├── table_1_structure.jpg       # Structure visualization
│   └── table_1_rows.jpg            # Rows visualization
├── page_2/
│   └── ...
└── ...
```

## How It Works

1. **PDF to Images**: Converts PDF pages to images using `pdf2image`
2. **Table Detection**: Detects tables using Table Transformer
3. **Structure Recognition**: Recognizes rows and columns in each table
4. **OCR**: Extracts text from cells using PaddleOCR
5. **CSV Generation**: Saves extracted data as CSV files

## Model Credits

This project uses the following models:

- [microsoft/table-transformer-detection](https://huggingface.co/microsoft/table-transformer-detection) for table detection
- [microsoft/table-structure-recognition-v1.1-all](https://huggingface.co/microsoft/table-structure-recognition-v1.1-all) for structure recognition
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for optical character recognition