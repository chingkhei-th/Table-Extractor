# PDF Table Extraction with OCR

A tool for extracting tables from PDF documents using image-based techniques and OCR.

## Features

- Loads PDF files and converts them to images
- Detects tables in PDF pages
- Crops tables from pages
- Identifies rows and columns in tables
- Extracts text using PaddleOCR
- Exports table data to CSV files
- Merges tables from multiple pages into a single CSV

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Note for Windows users

For Windows, you might need to install Poppler separately:

1. Download the Poppler binary from: https://github.com/oschwartz10612/poppler-windows/releases/
2. Extract it to a folder (e.g., C:\poppler)
3. Add the bin directory to your PATH: `C:\poppler\bin`

## Usage

Run the table extraction tool with:

```bash
python main.py path_to_your_pdf.pdf
```

### Optional arguments

- `--output_dir`, `-o`: Output directory (default: 'output')
- `--dpi`: DPI for PDF to image conversion (default: 300)
- `--denoise`: Apply denoising to images (default: True)
- `--no-denoise`: Disable denoising
- `--morph-close`: Apply morphological closing (default: True)
- `--no-morph-close`: Disable morphological closing
- `--crop-padding`: Padding around cropped tables (default: 1)
- `--lang`: Language for OCR (default: 'en')
- `--use-gpu`: Use GPU for OCR if available (default: False)

Example:

```bash
python main.py my_document.pdf --output_dir my_tables --dpi 400 --lang en --use-gpu
```

## Output

The tool creates the following directory structure:

```
output
├── merged_table.csv      # All tables merged into single CSV
├── cropped/              # Cropped table images
│   ├── page_0_table_0.png
│   ├── page_0_table_1.png
│   └── ...
├── csv/                  # Individual table CSV files
│   ├── page_0_table_0.csv
│   ├── page_0_table_1.csv
│   └── ...
├── detected/             # Pages with table detection boxes
│   ├── page_0.png
│   ├── page_1.png
│   └── ...
├── original/             # Original page images
│   ├── page_0.png
│   ├── page_1.png
│   └── ...
└── structure/            # Tables with grid structure
    ├── page_0_table_0.png
    ├── page_0_table_1.png
    └── ...
```

Each subdirectory contains specific information about the processing pipeline:

- **original/**: Original PDF pages as images
- **detected/**: Pages with green boxes showing detected table regions
- **cropped/**: Individual cropped tables from each page
- **structure/**: Tables with detected grid structure (red lines)
- **csv/**: CSV files containing the extracted text from each table
- **merged_table.csv**: All tables combined into a single CSV file, ordered by page and table number