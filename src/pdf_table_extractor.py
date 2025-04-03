import os
import re
import pandas as pd
from paddleocr import PPStructure, save_structure_res


# Step 1: Extract layout and tables from a PDF
def extract_layout_and_tables(pdf_path, save_folder):
    """
    Extracts layout and tables from a PDF using PaddleOCR's PP-StructureV2.

    Args:
        pdf_path (str): Path to the input PDF file.
        save_folder (str): Directory where extracted results will be saved.
    """
    ocr_engine = PPStructure(
        layout=True,
        table=True,
        ocr=True,
        show_log=False,
        structure_version="PP-StructureV2",
    )
    result = ocr_engine(pdf_path)
    for index, res in enumerate(result):
        save_structure_res(
            res, save_folder, os.path.basename(pdf_path).split(".")[0], index
        )
    print(f"Extracted layout and tables from {pdf_path} to {save_folder}")


# Step 2: Convert all .xlsx files to .csv
def convert_xlsx_to_csv(extracted_folder):
    """
    Converts all .xlsx files in the extracted folder to .csv.

    Args:
        extracted_folder (str): Directory containing the .xlsx files.
    """
    xlsx_files = [f for f in os.listdir(extracted_folder) if f.endswith(".xlsx")]
    print(f"Found {len(xlsx_files)} .xlsx files in {extracted_folder}")
    if not xlsx_files:
        print("No .xlsx files found in the extracted folder.")
        return

    for file_name in xlsx_files:
        file_path = os.path.join(extracted_folder, file_name)
        try:
            df = pd.read_excel(file_path)
            csv_file_path = file_path.replace(".xlsx", ".csv")
            df.to_csv(csv_file_path, index=False)
            print(f"Converted {file_name} to {os.path.basename(csv_file_path)}")
        except Exception as e:
            print(f"Error converting {file_name}: {e}")


# Step 3: Merge all .csv files into a single file
def merge_csv_files(extracted_folder, final_csv_path):
    """
    Merges all .csv files in the extracted folder into a single .csv file.

    Args:
        extracted_folder (str): Directory containing the .csv files.
        final_csv_path (str): Path where the merged .csv file will be saved.
    """

    def parse_file_name(file_name):
        match = re.search(r"\[(\d+), (\d+), (\d+), (\d+)\]_(\d+)\.csv", file_name)
        if match:
            x1, y1, x2, y2, page = map(int, match.groups())
            return page, x1, y1
        return None

    csv_files = [f for f in os.listdir(extracted_folder) if f.endswith(".csv")]
    print(f"Found {len(csv_files)} .csv files in {extracted_folder}")
    if not csv_files:
        print("No .csv files found to merge.")
        return

    file_info = []
    for file_name in csv_files:
        info = parse_file_name(file_name)
        if info:
            page, x1, y1 = info
            file_path = os.path.join(extracted_folder, file_name)
            file_info.append((page, y1, x1, file_path))
        else:
            print(f"Skipping file with unexpected name format: {file_name}")

    if not file_info:
        print("No .csv files matched the expected name format.")
        return

    file_info.sort(key=lambda x: (x[0], x[1], x[2]))  # Sort by page, y1, x1
    with open(final_csv_path, "w", encoding="utf-8") as f:
        for i, (page, y1, x1, file_path) in enumerate(file_info):
            try:
                df = pd.read_csv(file_path)
                f.write(f"Table from page {page}, position [{x1}, {y1}]\n")
                df.to_csv(f, header=True, index=False)
                if i < len(file_info) - 1:
                    f.write("\n")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    print(f"All .csv files have been merged into {final_csv_path}")


# Main function to run the process
def main(pdf_path, output_folder):
    """
    Runs the complete process: extraction, conversion, and merging.

    Args:
        pdf_path (str): Path to the input PDF file.
        output_folder (str): Base directory for output files.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define the subdirectory where extracted files are saved
    extracted_folder = os.path.join(
        output_folder, os.path.basename(pdf_path).split(".")[0]
    )

    # Step 1: Extract layout and tables
    extract_layout_and_tables(
        pdf_path, output_folder
    )  # Saves to output_folder/Test_statement

    # Step 2: Convert .xlsx to .csv in the extracted folder
    convert_xlsx_to_csv(extracted_folder)

    # Step 3: Merge .csv files into a single file in the output folder
    final_csv_path = os.path.join(output_folder, "merged_tables.csv")
    merge_csv_files(extracted_folder, final_csv_path)


# Example usage
if __name__ == "__main__":
    pdf_path = "../data/test-input/Test_statement.pdf"
    output_folder = "../data/output-v2/merge-csv"
    main(pdf_path, output_folder)
