"""File I/O utilities for table extraction."""

import os
import csv
import pandas as pd


def save_csv(data, output_path):
    """Save table data as CSV.

    Args:
        data (dict): Dictionary of row data.
        output_path (str): Path to save the CSV file.
    """
    with open(output_path, "w", newline="") as result_file:
        wr = csv.writer(result_file, dialect="excel")
        for row, row_text in data.items():
            wr.writerow(row_text)


def merge_csvs(all_data, output_dir, header_page=0):
    """Merge all extracted tables into a single CSV file.

    Args:
        all_data (list): List of table data from all pages.
        output_dir (str): Output directory to save the merged CSV.
        header_page (int): Index of the page to use for header.

    Returns:
        str: Path to the merged CSV file, or None if there's no data.
    """
    if not all_data:
        print("No data to merge")
        return None

    # Find the maximum number of columns across all tables
    max_columns = max(
        [cols for page in all_data for _, cols in page if cols > 0], default=0
    )

    merged_data = []

    # Keep track of the header row
    header = None

    # Process each page
    for page_idx, page_tables in enumerate(all_data):
        if not page_tables:
            continue

        for table_data, _ in page_tables:
            # Convert dict to list of rows
            rows = [table_data[row_idx] for row_idx in sorted(table_data.keys())]

            # For the first page's first table, save the header
            if page_idx == header_page and header is None and rows:
                header = rows[0]
                # Only add non-header rows from first page
                merged_data.extend(rows[1:])
            else:
                # For other pages, add all rows
                merged_data.extend(rows)

    # Ensure all rows have the same number of columns
    for i in range(len(merged_data)):
        if len(merged_data[i]) < max_columns:
            merged_data[i] = merged_data[i] + [""] * (max_columns - len(merged_data[i]))

    # Create the final DataFrame
    if header:
        # Ensure header has enough columns
        if len(header) < max_columns:
            header = header + [""] * (max_columns - len(header))

        # Create DataFrame with header
        df = pd.DataFrame(merged_data, columns=header)
    else:
        # Create DataFrame without header
        df = pd.DataFrame(merged_data)

    # Save merged CSV
    merged_csv_path = os.path.join(output_dir, "merged_tables.csv")
    df.to_csv(merged_csv_path, index=False)

    print(f"Merged CSV saved to {merged_csv_path}")
    return merged_csv_path
