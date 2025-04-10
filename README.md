# Table Extractor

A tool for extracting tables from PDF financial statements using deep learning models.

## Overview

This tool extracts tables from PDF documents using state-of-the-art deep learning models. It uses the Table Transformer (DETR) models from Microsoft for table detection and structure recognition, combined with OCR capabilities to extract tabular data from financial statements.

## Features

- Table detection in PDF documents
- Table structure recognition
- OCR for text extraction
- CSV output generation
- Visualization of detected tables and structures

## Installation

### Prerequisites

- Python 3.7+
- PyTorch
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository
2. Install dependencies:

    ```shell
    uv sync
    ```

    > Note: you'll need to install [`Astral UV`](https://docs.astral.sh/uv/getting-started/installation/) first if not already installed.

## Usage

### Command Line Interface

The tool can be run from the command line using the `run.py` script:

```shell
python run.py [options]
```

### Options

```shell
--pdf_path PATH             Path to the PDF file
--output DIR                Output directory (default: ../data/output/table-extractor)
--dpi DPI                   DPI for PDF to image conversion (default: 200)
--max-detection-size SIZE   Maximum size for detection model input (default: 800)
--max-structure-size SIZE   Maximum size for structure model input (default: 1000)
--crop-padding PADDING      Padding around table crops (default: 10)
--detection-model MODEL     Detection model name or path (default: microsoft/table-transformer-detection)
--structure-model MODEL     Structure model name or path (default: microsoft/table-structure-recognition-v1.1-all)
--detection-threshold FLOAT Detection confidence threshold (default: 0.5)
--ocr-lang LANG             OCR language (default: en)
```

### Example

```bash
python run.py --pdf_path ../data/financial_statement.pdf --output ../data/output/results --dpi 300
```

## Configuration

The tool uses a configuration class `TableExtractorConfig` that can be customized. Default values are:

- Input PDF path: `../data/test-input/Test_statement.pdf`
- Output directory: `../data/output/table-extractor`
- PDF DPI: 400
- Maximum detection size: 800
- Maximum structure size: 1000
- Crop padding: 10
- Detection model: `microsoft/table-transformer-detection`
- Structure model: `microsoft/table-structure-recognition-v1.1-all`
- OCR language: `en`
- Detection thresholds:
  - table: 0.5
  - table rotated: 0.5
  - no object: 10

## Output Structure

The tool creates the following output directories:

- `original/`: Original PDF pages as images
- `detected/`: Visualizations of detected tables
- `cropped/`: Cropped table images
- `structure/`: Visualizations of table structures
- `csv/`: Extracted table data in CSV format

## Architecture

The system consists of several components:

1. **TableExtractor**: Main class that orchestrates the extraction process
2. **TableDetectionModel**: Handles table detection in document images
3. **TableStructureModel**: Recognizes the structure of detected tables
4. **OCRModel**: Extracts text from table cells
5. **TemplateColumnDetector**: Detects columns in tables

## How It Works

1. **PDF to Images**: Converts PDF pages to images using `pdf2image`
2. **Table Detection**: Detects tables using Table Transformer
3. **Structure Recognition**: Recognizes rows and columns in each table
4. **OCR**: Extracts text from cells using PaddleOCR
5. **CSV Generation**: Saves extracted data as CSV files

## Model Credits

This project uses models from the [Microsoft Table Transformer (DETR)](https://github.com/microsoft/table-transformer) project and [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for optical character recognition.

Models with their task:

- [microsoft/table-transformer-detection](https://huggingface.co/microsoft/table-transformer-detection) for table detection.
- [microsoft/table-structure-recognition-v1.1-all](https://huggingface.co/microsoft/table-structure-recognition-v1.1-all) for structure recognition.
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for optical character recognition.
