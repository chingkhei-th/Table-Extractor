import os
import numpy as np
import torch
import csv
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
import io
import pandas as pd
from tqdm import tqdm
from paddleocr import PaddleOCR
from torchvision import transforms
from transformers import (
    AutoModelForObjectDetection,
    TableTransformerForObjectDetection
)
import pdf2image
import argparse

class MaxResize:
    """Resizes an image to a maximum size while maintaining aspect ratio."""
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))
        return resized_image

class TableExtractor:
    """Main class for extracting tables from PDF documents."""
    def __init__(self, output_dir="output"):
        """Initialize the table extractor with models and transforms."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Create output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "original"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "detected"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "cropped"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "structure"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "csv"), exist_ok=True)

        # Load table detection model
        print("Loading table detection model...")
        self.detection_model = AutoModelForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection",
            revision="no_timm"
        ).to(self.device)

        # Load table structure recognition model
        print("Loading table structure recognition model...")
        self.structure_model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-structure-recognition-v1.1-all"
        ).to(self.device)

        # Initialize OCR
        print("Initializing PaddleOCR...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

        # Set up transforms
        self.detection_transform = transforms.Compose([
            MaxResize(800),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.structure_transform = transforms.Compose([
            MaxResize(1000),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Set detection thresholds
        self.detection_class_thresholds = {
            "table": 0.5,
            "table rotated": 0.5,
            "no object": 10
        }

        # Update id2label for models
        self.detection_id2label = self.detection_model.config.id2label
        self.detection_id2label[len(self.detection_id2label)] = "no object"

        self.structure_id2label = self.structure_model.config.id2label
        self.structure_id2label[len(self.structure_id2label)] = "no object"

    def convert_pdf_to_images(self, pdf_path):
        """Convert PDF file to a list of PIL images."""
        print(f"Converting PDF to images: {pdf_path}")
        return pdf2image.convert_from_path(pdf_path)

    def box_cxcywh_to_xyxy(self, x):
        """Convert bounding box from center format to corner format."""
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        """Rescale bounding boxes according to image size."""
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def outputs_to_objects(self, outputs, img_size, id2label):
        """Convert model outputs to object list with labels, scores, and bounding boxes."""
        m = outputs.logits.softmax(-1).max(-1)
        pred_labels = list(m.indices.detach().cpu().numpy())[0]
        pred_scores = list(m.values.detach().cpu().numpy())[0]
        pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
        pred_bboxes = [elem.tolist() for elem in self.rescale_bboxes(pred_bboxes, img_size)]

        objects = []
        for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
            class_label = id2label[int(label)]
            if not class_label == 'no object':
                objects.append({
                    'label': class_label,
                    'score': float(score),
                    'bbox': [float(elem) for elem in bbox]
                })

        return objects

    def iob(self, bbox1, bbox2):
        """
        Compute the intersection over box area ratio between two bounding boxes.
        Used to determine if a token belongs to a table.
        """
        x_min = max(bbox1[0], bbox2[0])
        y_min = max(bbox1[1], bbox2[1])
        x_max = min(bbox1[2], bbox2[2])
        y_max = min(bbox1[3], bbox2[3])

        if x_max < x_min or y_max < y_min:
            return 0.0

        intersection_area = (x_max - x_min) * (y_max - y_min)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])

        if bbox1_area == 0:
            return 0.0

        return intersection_area / bbox1_area

    def objects_to_crops(self, img, objects, padding=10):
        """Process the bounding boxes to crop table images."""
        tokens = []  # Empty tokens list as we're not using token information here
        table_crops = []

        for obj in objects:
            if obj['score'] < self.detection_class_thresholds[obj['label']]:
                continue

            cropped_table = {}
            bbox = obj['bbox']
            bbox = [
                max(0, bbox[0]-padding),
                max(0, bbox[1]-padding),
                min(img.width, bbox[2]+padding),
                min(img.height, bbox[3]+padding)
            ]

            cropped_img = img.crop(bbox)
            table_tokens = [token for token in tokens if self.iob(token['bbox'], bbox) >= 0.5]

            for token in table_tokens:
                token['bbox'] = [
                    token['bbox'][0]-bbox[0],
                    token['bbox'][1]-bbox[1],
                    token['bbox'][2]-bbox[0],
                    token['bbox'][3]-bbox[1]
                ]

            # If table is predicted to be rotated, just keep original orientation as requested
            cropped_table['image'] = cropped_img
            cropped_table['tokens'] = table_tokens
            cropped_table['bbox'] = bbox
            cropped_table['label'] = obj['label']

            table_crops.append(cropped_table)

        return table_crops

    def fig2img(self, fig):
        """Convert a Matplotlib figure to a PIL Image."""
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img

    def visualize_detected_tables(self, img, det_tables):
        """Visualize detected tables on the image."""
        plt.figure(figsize=(20, 20))
        plt.imshow(img, interpolation="lanczos")
        ax = plt.gca()

        for det_table in det_tables:
            bbox = det_table['bbox']

            if det_table['label'] == 'table':
                facecolor = (1, 0, 0.45)
                edgecolor = (1, 0, 0.45)
                alpha = 0.3
                linewidth = 2
                hatch='//////'
            elif det_table['label'] == 'table rotated':
                facecolor = (0.95, 0.6, 0.1)
                edgecolor = (0.95, 0.6, 0.1)
                alpha = 0.3
                linewidth = 2
                hatch='//////'
            else:
                continue

            rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth,
                                    edgecolor='none', facecolor=facecolor, alpha=0.1)
            ax.add_patch(rect)
            rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth,
                                    edgecolor=edgecolor, facecolor='none', linestyle='-', alpha=alpha)
            ax.add_patch(rect)
            rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=0,
                                    edgecolor=edgecolor, facecolor='none', linestyle='-', hatch=hatch, alpha=0.2)
            ax.add_patch(rect)

        plt.xticks([], [])
        plt.yticks([], [])

        legend_elements = [
            Patch(facecolor=(1, 0, 0.45), edgecolor=(1, 0, 0.45),
                  label='Table', hatch='//////', alpha=0.3),
            Patch(facecolor=(0.95, 0.6, 0.1), edgecolor=(0.95, 0.6, 0.1),
                  label='Table (rotated)', hatch='//////', alpha=0.3)
        ]

        plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.02),
                 loc='upper center', borderaxespad=0, fontsize=10, ncol=2)
        plt.gcf().set_size_inches(10, 10)
        plt.axis('off')

        fig = plt.gcf()
        visualized_image = self.fig2img(fig)
        plt.close()

        return visualized_image

    def visualize_table_structure(self, cropped_table, cells):
        """Visualize the detected cells on the cropped table image."""
        cropped_table_visualized = cropped_table.copy()
        draw = ImageDraw.Draw(cropped_table_visualized)

        for cell in cells:
            draw.rectangle(cell["bbox"], outline="red")

        return cropped_table_visualized

    def get_cell_coordinates_by_row(self, table_data):
        """Extract row and column information to create cell coordinates."""
        # Extract rows and columns
        rows = [entry for entry in table_data if entry['label'] == 'table row']
        columns = [entry for entry in table_data if entry['label'] == 'table column']

        # Sort rows and columns by their Y and X coordinates
        rows.sort(key=lambda x: x['bbox'][1])
        columns.sort(key=lambda x: x['bbox'][0])

        # Generate cell coordinates and count cells in each row
        cell_coordinates = []

        for row in rows:
            row_cells = []
            for column in columns:
                cell_bbox = [
                    column['bbox'][0],
                    row['bbox'][1],
                    column['bbox'][2],
                    row['bbox'][3]
                ]
                row_cells.append({'column': column['bbox'], 'cell': cell_bbox})

            # Sort cells in the row by X coordinate
            row_cells.sort(key=lambda x: x['column'][0])

            # Append row information to cell_coordinates
            cell_coordinates.append({
                'row': row['bbox'],
                'cells': row_cells,
                'cell_count': len(row_cells)
            })

        # Sort rows from top to bottom
        cell_coordinates.sort(key=lambda x: x['row'][1])

        return cell_coordinates

    def apply_ocr(self, cropped_table, cell_coordinates):
        """Apply OCR to each cell in the table."""
        data = dict()
        max_num_columns = 0

        for idx, row in enumerate(cell_coordinates):
            row_text = []
            for cell in row["cells"]:
                # Crop cell out of image
                cell_image = np.array(cropped_table.crop(cell["cell"]))

                # Apply OCR
                result = self.ocr.ocr(cell_image, cls=True)

                if result and len(result) > 0 and result[0]:
                    # Extract text from results
                    text_parts = []
                    for line in result:
                        if line:
                            for item in line:
                                if len(item) >= 2:  # Make sure we have the text and confidence
                                    text_parts.append(item[1][0])  # item[1][0] contains the text
                    text = " ".join(text_parts)
                    row_text.append(text)
                else:
                    row_text.append("")  # Empty string for cells with no detected text

            if len(row_text) > max_num_columns:
                max_num_columns = len(row_text)

            data[idx] = row_text

        # Pad rows to ensure all have the same number of columns
        for row, row_data in data.copy().items():
            if len(row_data) != max_num_columns:
                row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
            data[row] = row_data

        return data, max_num_columns

    def save_csv(self, data, output_path):
        """Save table data as CSV."""
        with open(output_path, 'w', newline='') as result_file:
            wr = csv.writer(result_file, dialect='excel')
            for row, row_text in data.items():
                wr.writerow(row_text)

    def process_image(self, image, page_num):
        """Process a single page image to extract tables."""
        # Save original image
        orig_image_path = os.path.join(self.output_dir, "original", f"page_{page_num}.jpg")
        image.save(orig_image_path)

        # Table detection
        pixel_values = self.detection_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.detection_model(pixel_values)

        # Extract table objects
        objects = self.outputs_to_objects(outputs, image.size, self.detection_id2label)

        if not objects:
            print(f"No tables detected on page {page_num}")
            return None, None

        # Visualize detected tables
        detected_image = self.visualize_detected_tables(image, objects)
        detected_image = detected_image.convert("RGB")
        detected_image_path = os.path.join(self.output_dir, "detected", f"page_{page_num}_detected.jpg")
        detected_image.save(detected_image_path)

        # Process each detected table
        all_page_data = []
        max_columns = 0

        for table_idx, table_crop_info in enumerate(self.objects_to_crops(image, objects)):
            # Get the cropped table image
            cropped_table = table_crop_info['image']
            cropped_table_path = os.path.join(self.output_dir, "cropped", f"page_{page_num}_table_{table_idx}.jpg")
            cropped_table.save(cropped_table_path)

            # Apply structure recognition
            pixel_values = self.structure_transform(cropped_table).unsqueeze(0).to(self.device)

            with torch.no_grad():
                structure_outputs = self.structure_model(pixel_values)

            # Extract cell structure
            cells = self.outputs_to_objects(structure_outputs, cropped_table.size, self.structure_id2label)

            # Visualize structure
            structure_image = self.visualize_table_structure(cropped_table, cells)
            structure_image_path = os.path.join(self.output_dir, "structure", f"page_{page_num}_table_{table_idx}_structure.jpg")
            structure_image.save(structure_image_path)

            # Get cell coordinates
            cell_coordinates = self.get_cell_coordinates_by_row(cells)

            if not cell_coordinates:
                print(f"No rows/columns detected in table {table_idx} on page {page_num}")
                continue

            # Apply OCR
            table_data, table_columns = self.apply_ocr(cropped_table, cell_coordinates)

            # Update max columns if this table has more
            max_columns = max(max_columns, table_columns)

            # Save individual table CSV
            table_csv_path = os.path.join(self.output_dir, "csv", f"page_{page_num}_table_{table_idx}.csv")
            self.save_csv(table_data, table_csv_path)

            # Add to page data
            all_page_data.append((table_data, table_columns))

        return all_page_data, max_columns

    def merge_csvs(self, all_data, header_page=0):
        """Merge all extracted tables into a single CSV file."""
        if not all_data:
            print("No data to merge")
            return

        # Find the maximum number of columns across all tables
        max_columns = max([cols for page in all_data for _, cols in page if cols > 0], default=0)

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
        merged_csv_path = os.path.join(self.output_dir, "merged_tables.csv")
        df.to_csv(merged_csv_path, index=False)

        print(f"Merged CSV saved to {merged_csv_path}")
        return merged_csv_path

    #  template-based column detection
    def process_pdf(self, pdf_path):
        """Process a PDF file with improved column detection."""
        # Convert PDF to images
        images = self.convert_pdf_to_images(pdf_path)

        # First, process the first page to get column structure
        print(f"\nProcessing first page to extract column structure template")
        first_page = images[0]

        # Extract tables from first page
        pixel_values = self.detection_transform(first_page).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.detection_model(pixel_values)
        first_page_objects = self.outputs_to_objects(outputs, first_page.size, self.detection_id2label)

        # If no tables on first page, use normal processing
        if not first_page_objects:
            print("No tables on first page. Falling back to regular processing.")
            return self.process_pdf(pdf_path)

        # Get first page table
        first_page_tables = self.objects_to_crops(first_page, first_page_objects)
        if not first_page_tables:
            return self.process_pdf(pdf_path)

        first_table = first_page_tables[0]['image']

        # Extract column structure from first page
        template_columns = self.get_template_columns(first_table)

        if not template_columns:
            print("Could not extract template columns. Using whitespace analysis.")
            template_columns = self.detect_columns_by_whitespace(first_table)

        # Process each page
        all_data = []

        for page_idx, image in enumerate(tqdm(images, desc="Processing pages")):
            page_num = page_idx + 1
            print(f"\nProcessing page {page_num}/{len(images)}")

            # Detect tables on this page
            pixel_values = self.detection_transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.detection_model(pixel_values)
            objects = self.outputs_to_objects(outputs, image.size, self.detection_id2label)

            # Save original image
            orig_image_path = os.path.join(self.output_dir, "original", f"page_{page_num}.jpg")
            image.save(orig_image_path)

            if not objects:
                print(f"No tables detected on page {page_num}")
                all_data.append(None)
                continue

            # Visualize detected tables
            detected_image = self.visualize_detected_tables(image, objects)
            detected_image_path = os.path.join(self.output_dir, "detected", f"page_{page_num}_detected.jpg")
            detected_image.save(detected_image_path)

            # Process tables on this page
            page_data = []

            for table_idx, table_crop_info in enumerate(self.objects_to_crops(image, objects)):
                cropped_table = table_crop_info['image']
                cropped_table_path = os.path.join(self.output_dir, "cropped", f"page_{page_num}_table_{table_idx}.jpg")
                cropped_table.save(cropped_table_path)

                # For first page, use structure model results
                if page_idx == 0:
                    # Use structure model to detect both rows and columns
                    pixel_values = self.structure_transform(cropped_table).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        structure_outputs = self.structure_model(pixel_values)
                    cells = self.outputs_to_objects(structure_outputs, cropped_table.size, self.structure_id2label)
                else:
                    # For subsequent pages, use structure model only for rows
                    pixel_values = self.structure_transform(cropped_table).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        structure_outputs = self.structure_model(pixel_values)
                    cells = self.outputs_to_objects(structure_outputs, cropped_table.size, self.structure_id2label)

                    # Extract rows
                    rows = [entry for entry in cells if entry['label'] == 'table row']

                    # Replace detected columns with template columns adjusted to this table's size
                    cols = self.apply_template_columns(template_columns, cropped_table.size)

                    # Remove existing columns from cells
                    cells = [cell for cell in cells if cell['label'] != 'table column']

                    # Add template columns to cells
                    cells.extend(cols)

                # Post-process columns to fix overlaps
                columns = [cell for cell in cells if cell['label'] == 'table column']
                processed_columns = self.post_process_columns(columns)

                # Replace columns in cells with processed ones
                cells = [cell for cell in cells if cell['label'] != 'table column']
                cells.extend(processed_columns)

                # Visualize structure with processed columns
                structure_image = self.visualize_table_structure(cropped_table, cells)
                structure_image_path = os.path.join(self.output_dir, "structure", f"page_{page_num}_table_{table_idx}_structure.jpg")
                structure_image.save(structure_image_path)

                # Get cell coordinates
                cell_coordinates = self.get_cell_coordinates_by_row(cells)

                if not cell_coordinates:
                    print(f"No rows/columns detected in table {table_idx} on page {page_num}")
                    continue

                # Apply OCR
                table_data, table_columns = self.apply_ocr(cropped_table, cell_coordinates)

                # Save individual table CSV
                table_csv_path = os.path.join(self.output_dir, "csv", f"page_{page_num}_table_{table_idx}.csv")
                self.save_csv(table_data, table_csv_path)

                # Add to page data
                page_data.append((table_data, table_columns))

            all_data.append(page_data)

        # Merge all tables into a single CSV
        if any(all_data):
            merged_csv = self.merge_csvs(all_data, header_page=0)
            print(f"Processing complete. Results saved to {self.output_dir}")
            return merged_csv
        else:
            print("No tables were found in the PDF.")
            return None

    def extract_column_template(self, image):
        """Extract column structure from the first page to use as a template."""
        # Table detection
        pixel_values = self.detection_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.detection_model(pixel_values)

        # Extract table objects
        objects = self.outputs_to_objects(outputs, image.size, self.detection_id2label)

        if not objects:
            print("No tables detected on first page for template")
            return None

        # Get the first table
        table_crops = self.objects_to_crops(image, objects)
        if not table_crops:
            return None

        # Get the cropped table image
        cropped_table = table_crops[0]['image']

        # Apply structure recognition
        pixel_values = self.structure_transform(cropped_table).unsqueeze(0).to(self.device)

        with torch.no_grad():
            structure_outputs = self.structure_model(pixel_values)

        # Extract cell structure
        cells = self.outputs_to_objects(structure_outputs, cropped_table.size, self.structure_id2label)

        # Extract just the columns and normalize their positions
        columns = [entry for entry in cells if entry['label'] == 'table column']

        if not columns:
            print("No columns detected in the first page table")
            return None

        # Sort columns by x-coordinate
        columns.sort(key=lambda x: x['bbox'][0])

        # Get table width
        table_width = cropped_table.width

        # Normalize column positions (as ratios of table width)
        normalized_columns = []
        for col in columns:
            normalized_col = {
                'x_start_ratio': col['bbox'][0] / table_width,
                'x_end_ratio': col['bbox'][2] / table_width,
                'original_bbox': col['bbox']
            }
            normalized_columns.append(normalized_col)

        return normalized_columns

    def process_image_with_template(self, image, page_num, template_columns):
        """Process a single page image using column template."""
        if template_columns is None:
            print(f"No column template available for page {page_num}")
            return self.process_image(image, page_num)[0]  # Fall back to regular processing

        # Save original image
        orig_image_path = os.path.join(self.output_dir, "original", f"page_{page_num}.jpg")
        image.save(orig_image_path)

        # Table detection
        pixel_values = self.detection_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.detection_model(pixel_values)

        # Extract table objects
        objects = self.outputs_to_objects(outputs, image.size, self.detection_id2label)

        if not objects:
            print(f"No tables detected on page {page_num}")
            return None

        # Visualize detected tables
        detected_image = self.visualize_detected_tables(image, objects)
        detected_image = detected_image.convert("RGB")
        detected_image_path = os.path.join(self.output_dir, "detected", f"page_{page_num}_detected.jpg")
        detected_image.save(detected_image_path)

        # Process each detected table
        all_page_data = []

        for table_idx, table_crop_info in enumerate(self.objects_to_crops(image, objects)):
            # Get the cropped table image
            cropped_table = table_crop_info['image']
            cropped_table_path = os.path.join(self.output_dir, "cropped", f"page_{page_num}_table_{table_idx}.jpg")
            cropped_table.save(cropped_table_path)

            # Apply structure recognition just to get rows
            pixel_values = self.structure_transform(cropped_table).unsqueeze(0).to(self.device)

            with torch.no_grad():
                structure_outputs = self.structure_model(pixel_values)

            # Extract cell structure
            cells = self.outputs_to_objects(structure_outputs, cropped_table.size, self.structure_id2label)

            # Extract rows
            rows = [entry for entry in cells if entry['label'] == 'table row']

            # Sort rows by Y coordinate
            rows.sort(key=lambda x: x['bbox'][1])

            # Apply template columns to this table
            table_width = cropped_table.width
            template_applied_cells = []

            # Add rows to cells
            for row in rows:
                template_applied_cells.append(row)

            # Create column objects from template
            for col in template_columns:
                x_start = col['x_start_ratio'] * table_width
                x_end = col['x_end_ratio'] * table_width

                # Create column object
                column_obj = {
                    'label': 'table column',
                    'score': 0.99,  # High confidence since it's from template
                    'bbox': [x_start, 0, x_end, cropped_table.height]
                }

                template_applied_cells.append(column_obj)

            # Visualize structure with template columns
            structure_image = self.visualize_table_structure(cropped_table, template_applied_cells)
            structure_image_path = os.path.join(self.output_dir, "structure", f"page_{page_num}_table_{table_idx}_structure.jpg")
            structure_image.save(structure_image_path)

            # Get cell coordinates using the template columns
            cell_coordinates = self.get_cell_coordinates_by_row(template_applied_cells)

            if not cell_coordinates:
                print(f"No rows/columns detected in table {table_idx} on page {page_num}")
                continue

            # Apply OCR
            table_data, table_columns = self.apply_ocr(cropped_table, cell_coordinates)

            # Save individual table CSV
            table_csv_path = os.path.join(self.output_dir, "csv", f"page_{page_num}_table_{table_idx}.csv")
            self.save_csv(table_data, table_csv_path)

            # Add to page data
            all_page_data.append((table_data, table_columns))

        return all_page_data

    # ==== Post-Processing Column Detection ====
    def post_process_columns(self, columns, min_gap=5):
        """
        Post-process detected columns to fix overlaps and maintain consistency.

        Args:
            columns: List of column objects with bboxes
            min_gap: Minimum gap between columns in pixels

        Returns:
            List of corrected column objects
        """
        if not columns or len(columns) <= 1:
            return columns

        # Sort columns by x-coordinate
        columns.sort(key=lambda x: x['bbox'][0])

        # Fix overlapping columns
        corrected_columns = [columns[0]]

        for i in range(1, len(columns)):
            current_col = columns[i]
            previous_col = corrected_columns[-1]

            # Check for overlap
            if current_col['bbox'][0] <= previous_col['bbox'][2] + min_gap:
                # Take the average of the overlapping boundaries
                midpoint = (previous_col['bbox'][2] + current_col['bbox'][0]) / 2

                # Update previous column's right boundary
                previous_col['bbox'][2] = midpoint - min_gap/2

                # Update current column's left boundary
                current_col['bbox'][0] = midpoint + min_gap/2

            corrected_columns.append(current_col)

        return corrected_columns

    def filter_duplicate_columns(self, columns, similarity_threshold=0.9):
        """
        Filter out duplicate columns that are very similar to each other.

        Args:
            columns: List of column objects with bboxes
            similarity_threshold: Threshold for considering columns as duplicates

        Returns:
            List of filtered column objects
        """
        if not columns or len(columns) <= 1:
            return columns

        # Sort columns by x-coordinate
        columns.sort(key=lambda x: x['bbox'][0])

        filtered_columns = [columns[0]]

        for i in range(1, len(columns)):
            current_col = columns[i]
            previous_col = filtered_columns[-1]

            # Calculate overlap percentage
            x_overlap = min(current_col['bbox'][2], previous_col['bbox'][2]) - max(current_col['bbox'][0], previous_col['bbox'][0])

            if x_overlap <= 0:
                # No overlap, add the column
                filtered_columns.append(current_col)
                continue

            # Calculate overlap ratio relative to the smaller column width
            width1 = previous_col['bbox'][2] - previous_col['bbox'][0]
            width2 = current_col['bbox'][2] - current_col['bbox'][0]
            min_width = min(width1, width2)

            overlap_ratio = x_overlap / min_width

            # If overlap ratio is above threshold, consider as duplicate
            if overlap_ratio < similarity_threshold:
                filtered_columns.append(current_col)

        return filtered_columns

    def apply_column_consistency(self, pages_columns):
        """
        Apply consistency to columns across pages by clustering column positions.

        Args:
            pages_columns: List of columns from each page

        Returns:
            Consistent columns to use across all pages
        """
        import numpy as np
        from sklearn.cluster import DBSCAN

        # Extract column boundaries from all pages
        all_left_boundaries = []
        all_right_boundaries = []

        for page_cols in pages_columns:
            if not page_cols:
                continue

            for col in page_cols:
                all_left_boundaries.append(col['bbox'][0])
                all_right_boundaries.append(col['bbox'][2])

        # If no columns found, return empty list
        if not all_left_boundaries:
            return []

        # Cluster left boundaries
        left_boundaries = np.array(all_left_boundaries).reshape(-1, 1)
        left_clustering = DBSCAN(eps=20, min_samples=2).fit(left_boundaries)
        left_labels = left_clustering.labels_

        # Cluster right boundaries
        right_boundaries = np.array(all_right_boundaries).reshape(-1, 1)
        right_clustering = DBSCAN(eps=20, min_samples=2).fit(right_boundaries)
        right_labels = right_clustering.labels_

        # Get cluster centers for left boundaries
        unique_left_labels = set(left_labels)
        left_centers = []

        for label in unique_left_labels:
            if label == -1:  # Skip noise
                continue
            mask = left_labels == label
            center = np.mean(left_boundaries[mask])
            left_centers.append(center)

        # Get cluster centers for right boundaries
        unique_right_labels = set(right_labels)
        right_centers = []

        for label in unique_right_labels:
            if label == -1:  # Skip noise
                continue
            mask = right_labels == label
            center = np.mean(right_boundaries[mask])
            right_centers.append(center)

        # Sort centers
        left_centers.sort()
        right_centers.sort()

        # Create consistent columns
        consistent_columns = []

        # Handle case where we have different numbers of left and right boundaries
        min_length = min(len(left_centers), len(right_centers))

        for i in range(min_length):
            consistent_columns.append({
                'label': 'table column',
                'score': 0.99,
                'bbox': [left_centers[i], 0, right_centers[i], 1000]  # Height will be adjusted later
            })

        return consistent_columns
    # ========

    # ==== Rule-based column detection
    def detect_columns_by_whitespace(self, image, min_gap_width=10, min_column_width=20):
        """
        Detect table columns using whitespace analysis.

        Args:
            image: PIL Image of the table
            min_gap_width: Minimum width of whitespace to consider as column separator
            min_column_width: Minimum width of a column

        Returns:
            List of column objects with bboxes
        """
        # Convert image to grayscale numpy array
        img_array = np.array(image.convert('L'))

        # Get image height and width
        height, width = img_array.shape

        # Calculate vertical projection (sum along y-axis)
        # Higher values indicate more black pixels (text)
        vertical_projection = np.sum(255 - img_array, axis=0)

        # Normalize projection
        if np.max(vertical_projection) > 0:
            vertical_projection = vertical_projection / np.max(vertical_projection)

        # Find column separators (gaps with minimal text)
        gaps = []
        in_gap = False
        gap_start = 0

        for x in range(width):
            if vertical_projection[x] < 0.05:  # Threshold for whitespace
                if not in_gap:
                    in_gap = True
                    gap_start = x
            else:
                if in_gap:
                    in_gap = False
                    gap_end = x
                    if gap_end - gap_start >= min_gap_width:
                        gaps.append((gap_start, gap_end))

        # Add final gap if needed
        if in_gap:
            gaps.append((gap_start, width))

        # Convert gaps to columns
        columns = []

        # Handle the case when no gaps are found
        if not gaps:
            return [{
                'label': 'table column',
                'score': 0.99,
                'bbox': [0, 0, width, height]
            }]

        # First column starts at x=0
        col_start = 0

        for gap_start, gap_end in gaps:
            # Column ends at the start of the gap
            col_end = gap_start

            # Check if column is wide enough
            if col_end - col_start >= min_column_width:
                columns.append({
                    'label': 'table column',
                    'score': 0.99,
                    'bbox': [col_start, 0, col_end, height]
                })

            # Next column starts at the end of the gap
            col_start = gap_end

        # Add the last column if needed
        if col_start < width and width - col_start >= min_column_width:
            columns.append({
                'label': 'table column',
                'score': 0.99,
                'bbox': [col_start, 0, width, height]
            })

        return columns

    def detect_columns_by_text_alignment(self, image, ocr_results):
        """
        Detect table columns by analyzing text alignment patterns.

        Args:
            image: PIL Image of the table
            ocr_results: Text detection results from OCR

        Returns:
            List of column objects with bboxes
        """
        # Extract left and right text boundaries
        left_bounds = []
        right_bounds = []

        for result in ocr_results:
            # Extract bounding boxes of text from OCR results
            # This depends on the specific OCR output format
            for line in result:
                if line:
                    for item in line:
                        if len(item) >= 2:
                            # Get coordinates - format depends on PaddleOCR output
                            coords = item[0]
                            # Find leftmost and rightmost x coordinates
                            x_coords = [point[0] for point in coords]
                            left_bounds.append(min(x_coords))
                            right_bounds.append(max(x_coords))

        # If no text found, return empty
        if not left_bounds:
            return []

        # Cluster left bounds to find column starts
        from sklearn.cluster import DBSCAN

        # Convert to numpy arrays for clustering
        left_array = np.array(left_bounds).reshape(-1, 1)

        # Perform clustering
        clustering = DBSCAN(eps=20, min_samples=2).fit(left_array)
        labels = clustering.labels_

        # Extract cluster centers
        unique_labels = set(labels)
        centers = []

        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
            mask = labels == label
            center = np.mean(left_array[mask])
            centers.append(center)

        # Sort centers
        centers.sort()

        # Create column objects
        columns = []
        height = image.height

        # Handle first column
        if centers:
            if centers[0] > 20:  # If first center is not at the left edge
                columns.append({
                    'label': 'table column',
                    'score': 0.99,
                    'bbox': [0, 0, centers[0] - 10, height]
                })

        # Create columns between centers
        for i in range(len(centers) - 1):
            columns.append({
                'label': 'table column',
                'score': 0.99,
                'bbox': [centers[i], 0, centers[i+1] - 10, height]
            })

        # Add last column
        if centers:
            columns.append({
                'label': 'table column',
                'score': 0.99,
                'bbox': [centers[-1], 0, image.width, height]
            })

        return columns
    # ========

def main():
    parser = argparse.ArgumentParser(description='Extract tables from PDF documents')
    parser.add_argument('pdf_path', type=str, help='Path to the PDF file', default='../data/test-input/Test_statement.pdf')
    parser.add_argument(
        "--output",
        type=str,
        default="../data/output/table-transformer-v4",
        help="Output directory",
    )

    args = parser.parse_args()

    extractor = TableExtractor(output_dir=args.output)
    extractor.process_pdf(args.pdf_path)


if __name__ == "__main__":
    main()
