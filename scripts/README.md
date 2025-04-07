# PDF Table Extraction and Merging Script

This script extracts tables from a PDF file using [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) with PP-StructureV2, converts the extracted tables from `.xlsx` to `.csv` format, and merges all the `.csv` files into a single `.csv` file. It is designed to handle PDFs with multiple pages and multiple tables per page, making it ideal for processing documents like financial statements, reports, or invoices.

## Features
- **Table Extraction**: Uses PaddleOCR's PP-StructureV2 for accurate table detection and extraction.
- **Format Conversion**: Converts extracted tables from `.xlsx` to `.csv` for easier data manipulation.
- **Merging**: Combines all extracted tables into a single `.csv` file, preserving the order of pages and table positions.
- **Structured Output**: Saves results in a clear directory structure for easy access.

## Requirements
To run this script, you need the following Python packages:
- `paddlepaddle`
- `paddleocr`
- `pandas`
- `openpyxl`

## Installation
Install the required dependencies using `pip`:

```bash
uv sync
```

## Usage

**Prepare Your PDF:** Ensure you have a PDF file ready for processing.
**Update Paths:** Modify the pdf_path and output_folder variables in the script to point to your PDF file and the desired output directory.
**Run the Script:** Execute the script using Python:
```shell
python src/pdf_table_extractor.py
```
or
```shell
uv run src/pdf_table_extractor.py
```

**Review Results:** The script will create a directory structure under the specified output_folder, containing the extracted tables in `.csv` format. The final merged `.csv` file will be saved in the output_folder as well.

## Output Structure
The script organizes the output as follows:
```shell
output_folder/
    ├── <pdf_name>/
    │   ├── [x1, y1, x2, y2]_page.xlsx  # Extracted table in Excel format
    │   ├── [x1, y1, x2, y2]_page.csv   # Converted table in CSV format
    │   └── ...                         # Additional tables
    └── merged_tables.csv               # Merged CSV containing all tables
```

- `<pdf_name>/`: A subdirectory named after the input PDF (e.g., `Test_statement/`), containing individual tables in both `.xlsx` and `.csv` formats. Each file is named based on its position (`[x1, y1, x2, y2]`) and page number.
- `merged_tables.csv`: A single CSV file in the `output_folder`, containing all extracted tables ordered by page number and position. Each table is preceded by a line indicating its source page and position (e.g., "Table from page 0, position [163, 138]").

## How It Works
1. **Extraction:** The script uses PaddleOCR's PP-StructureV2 to analyze the PDF and extract tables, saving them as `.xlsx` files in a subdirectory named after the PDF.
2. **Conversion:** Each `.xlsx` file is converted to `.csv` format and saved in the same subdirectory.
3. **Merging:** All `.csv` files are merged into a single `merged_tables.csv` file, with each table’s data preceded by a descriptive line for easy identification.