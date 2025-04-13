# PDF Table Extraction with OCR

A tool for extracting tables from PDF documents using image-based techniques and OCR.

## Features

- Loads PDF files and converts them to images
- Detects tables in PDF pages using computer vision
- Crops tables from pages with configurable padding
- Identifies rows and columns using line detection
- Extracts text using PaddleOCR with multi-language support
- Exports table data to CSV files
- Merges tables with identical structures into consolidated CSVs
- Progress tracking with visual feedback
- Comprehensive logging of operations

## Key Features

### Intelligent CSV Merging
- Tables are grouped by column structure
- Multiple merged files created for different table formats
- Header row deduplication during merging
- Preserves original table order within structure groups

### Logging
- Detailed operations log in `table_extractor.log`
- Includes both application and OCR engine logs
- Overwritten for each new execution

### Progress Tracking
- Real-time progress bars for:
  - PDF loading
  - Table detection
  - OCR processing
  - File saving
- Clean completion message with timestamp

## Output Notes

- Tables with different column counts are saved in separate merged files
- First table's header is used as reference for merging
- Subsequent tables with matching headers are merged without header row
- Tables with unique headers are preserved as-is
- Empty rows/columns are automatically trimmed from CSV outputs

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```
or Using `Astral UV`
```shell
uv sync
```
> Note: you'll need to install [`Astral UV`](https://docs.astral.sh/uv/getting-started/installation/) first if not already installed.

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

- `--output_dir`, `-o`: Output directory (default: 'data/output')
- `--dpi`: DPI for PDF to image conversion (default: 300)
- `--denoise`/`--no-denoise`: Toggle image denoising (default: True)
- `--morph-close`/`--no-morph-close`: Toggle morphological closing (default: True)
- `--crop-padding`: Padding around cropped tables in pixels (default: 1)
- `--lang`: OCR language code (default: 'en')
- `--use-gpu`: Use GPU acceleration for OCR (requires compatible hardware)

Example:

```bash
python main.py financial_report.pdf --output_dir results --dpi 400 --lang en --use-gpu
```

## Output Structure

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

## Acknowledgement

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for optical character recognition.