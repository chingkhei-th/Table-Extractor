# PDF Table Extraction

A tool for extracting tables from PDF documents using image-based techniques.

## Features

- Loads PDF files and converts them to images
- Detects tables in PDF pages
- Crops tables from pages
- Identifies rows and columns in tables
- Outputs visualized table grids

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
- `--crop-padding`: Padding around cropped tables (default: 10)

Example:

```bash
python main.py my_document.pdf --output_dir my_tables --dpi 400
```

## Output

The tool saves the detected tables as PNG images in the specified output directory. Each file is named according to its page number and table index:

```
page_0_table_0.png
page_0_table_1.png
page_1_table_0.png
...
```

The output images show the original table with detected grid lines overlaid in red.