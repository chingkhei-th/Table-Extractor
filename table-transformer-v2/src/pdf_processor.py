import os
import csv
import logging
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from pdf2image import convert_from_path
from torchvision import transforms
import torch

from src.table_detection import TableDetector
from src.table_recognition import TableRecognizer
from src.ocr_processing import OCRProcessor
from src.utils import fig2img, iob, MaxResize
from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        self.detector = TableDetector()
        self.recognizer = TableRecognizer()
        self.ocr = OCRProcessor()
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    def process_pdf(self, pdf_path):
        logger.info(f"Processing PDF: {pdf_path}")
        try:
            images = convert_from_path(pdf_path, dpi=Config.DPI)
            logger.info(f"Converted {len(images)} pages to images")
        except Exception as e:
            logger.error(f"PDF conversion failed: {str(e)}")
            return None

        all_csvs = []

        for page_num, image in enumerate(images, start=1):
            page_dir = os.path.join(Config.OUTPUT_DIR, f"page_{page_num}")
            os.makedirs(page_dir, exist_ok=True)
            logger.info(f"Processing page {page_num}")

            # Save original page image
            image_path = os.path.join(page_dir, "original.jpg")
            image.save(image_path)

            # Detect tables
            try:
                outputs = self.detector.detect(image)
                objects = self._outputs_to_objects(outputs, image.size)
                logger.info(f"Detected {len(objects)} table objects")

                if not objects:
                    logger.warning(f"No tables detected in page {page_num}")
                    continue

                # Save table detection visualization
                self._save_visualization(image, objects, page_dir)

                # Crop table (single return value)
                cropped_table = self._crop_table(image, objects, page_dir)
                if cropped_table is None:
                    logger.warning(f"No table cropped from page {page_num}")
                    continue

                # Recognize table structure
                structure_outputs = self.recognizer.recognize(cropped_table)
                cells = self._outputs_to_objects(
                    structure_outputs,
                    cropped_table.size,
                    self.recognizer.model.config.id2label
                )

                # # Adjust coordinates if rotated
                # if is_rotated:
                #     cells = self._adjust_rotated_coordinates(cells, cropped_table.size)

                logger.info(f"Detected {len(cells)} table cells")

                # Save structure visualization
                self._save_structure_visualization(cropped_table, cells, page_dir)

                # Get cell coordinates
                cell_coords = self._get_cell_coordinates(cells)

                # Process OCR and save CSV
                csv_path = self._process_ocr(cropped_table, cell_coords, page_num, page_dir)
                all_csvs.append(csv_path)

            except Exception as e:
                logger.error(f"Error processing page {page_num}: {str(e)}")
                continue

        return self._merge_csvs(all_csvs)

    def _adjust_rotated_coordinates(self, cells, image_size):
        """Adjust coordinates for rotated tables"""
        adjusted_cells = []
        img_width, img_height = image_size

        for cell in cells:
            # Original coordinates
            xmin, ymin, xmax, ymax = cell["bbox"]

            # Rotate coordinates back 90 degrees (equivalent to rotating image -270)
            new_xmin = ymin
            new_ymin = img_width - xmax
            new_xmax = ymax
            new_ymax = img_width - xmin

            adjusted_cell = cell.copy()
            adjusted_cell["bbox"] = [new_xmin, new_ymin, new_xmax, new_ymax]
            adjusted_cells.append(adjusted_cell)

        return adjusted_cells

    def _outputs_to_objects(self, outputs, img_size, id2label=None):
        """Convert model outputs to detected objects dictionary"""
        def box_cxcywh_to_xyxy(x):
            x_c, y_c, w, h = x.unbind(-1)
            return torch.stack([x_c - 0.5 * w, y_c - 0.5 * h,
                              x_c + 0.5 * w, y_c + 0.5 * h], dim=1)

        def rescale_bboxes(out_bbox, size):
            return box_cxcywh_to_xyxy(out_bbox) * torch.tensor(
                [size[0], size[1], size[0], size[1]], dtype=torch.float32)

        if id2label is None:
            id2label = self.detector.model.config.id2label
        id2label = {int(k): v for k, v in id2label.items()}
        id2label[len(id2label)] = "no object"

        prob = outputs.logits.softmax(-1)
        scores, labels = prob.max(-1)
        scores = scores[0].detach().cpu().numpy()
        labels = labels[0].detach().cpu().numpy()
        bboxes = rescale_bboxes(outputs.pred_boxes[0].detach().cpu(), img_size).numpy()

        objects = []
        for score, label, bbox in zip(scores, labels, bboxes):
            class_label = id2label[int(label)]
            if class_label == "no object" or score < Config.DETECTION_THRESHOLDS.get(class_label, 0.5):
                continue
            objects.append({
                "label": class_label,
                "score": float(score),
                "bbox": [float(x) for x in bbox]
            })

        return objects

    def _save_visualization(self, image, objects, page_dir):
        """Save visualization with hatched patterns matching notebook cell 8"""
        plt.imshow(image)
        fig = plt.gcf()
        fig.set_size_inches(20, 20)
        ax = plt.gca()

        for obj in objects:
            bbox = obj["bbox"]
            label = obj["label"]

            if label == 'table':
                facecolor = (1, 0, 0.45)
                edgecolor = (1, 0, 0.45)
                alpha = 0.3
                hatch = '//////'
            elif label == 'table rotated':
                facecolor = (0.95, 0.6, 0.1)
                edgecolor = (0.95, 0.6, 0.1)
                alpha = 0.3
                hatch = '//////'
            else:
                continue

            # Create three overlapping rectangles like in notebook
            rect = Rectangle(
                (bbox[0], bbox[1]),
                bbox[2]-bbox[0],
                bbox[3]-bbox[1],
                linewidth=2,
                edgecolor='none',
                facecolor=facecolor,
                alpha=0.1
            )
            ax.add_patch(rect)

            rect = Rectangle(
                (bbox[0], bbox[1]),
                bbox[2]-bbox[0],
                bbox[3]-bbox[1],
                linewidth=2,
                edgecolor=edgecolor,
                facecolor='none',
                alpha=alpha
            )
            ax.add_patch(rect)

            rect = Rectangle(
                (bbox[0], bbox[1]),
                bbox[2]-bbox[0],
                bbox[3]-bbox[1],
                linewidth=0,
                edgecolor=edgecolor,
                facecolor='none',
                hatch=hatch,
                alpha=0.2
            )
            ax.add_patch(rect)

        # Add legend
        legend_elements = [
            Patch(facecolor=(1, 0, 0.45), edgecolor=(1, 0, 0.45),
                label='Table', hatch='//////', alpha=0.3),
            Patch(facecolor=(0.95, 0.6, 0.1), edgecolor=(0.95, 0.6, 0.1),
                label='Table (rotated)', hatch='//////', alpha=0.3)
        ]
        plt.legend(handles=legend_elements,
                   bbox_to_anchor=(0.5, -0.02),
                   loc='upper center',
                   borderaxespad=0,
                   fontsize=10,
                   ncol=2)

        plt.axis('off')
        viz_path = os.path.join(page_dir, "detected_tables.jpg")
        plt.savefig(viz_path, bbox_inches='tight', dpi=150)
        plt.close()

    def _save_structure_visualization(self, cropped_table, cells, page_dir):
        """Save table structure visualization matching notebook cell 17"""
        plt.figure(figsize=(16, 10))
        plt.imshow(cropped_table)
        ax = plt.gca()

        for cell in cells:
            if cell["label"] not in ["table row", "table column"]:
                continue

            bbox = cell["bbox"]
            score = cell["score"]
            label = cell["label"]

            xmin, ymin, xmax, ymax = bbox
            rect = Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                color="red" if label == "table row" else "blue",
                linewidth=3,
            )
            ax.add_patch(rect)

            text = f"{label}: {score:.2f}"
            ax.text(xmin, ymin, text, fontsize=10, bbox=dict(facecolor="yellow", alpha=0.5))

        plt.axis("off")
        structure_viz_path = os.path.join(page_dir, "detected_structure.jpg")
        plt.savefig(structure_viz_path, bbox_inches="tight", dpi=150)
        plt.close()

    def _crop_table(self, image, objects, page_dir):
        """Crop table without rotation (returns single Image object)"""
        try:
            table_objects = [obj for obj in objects if obj["label"] in ["table", "table rotated"]]
            if not table_objects:
                return None

            # Take first detected table
            table = max(table_objects, key=lambda x: x["score"])
            bbox = table["bbox"]

            # Add padding
            padding = Config.CROP_PADDING
            bbox = [
                max(0, bbox[0] - padding),
                max(0, bbox[1] - padding),
                min(image.width, bbox[2] + padding),
                min(image.height, bbox[3] + padding)
            ]

            cropped = image.crop(bbox)
            crop_path = os.path.join(page_dir, "cropped_table.jpg")
            cropped.save(crop_path)
            return cropped

        except Exception as e:
            logger.error(f"Table cropping failed: {str(e)}")
            return None

    def _get_cell_coordinates(self, cells):
        """Get cell coordinates without rotation adjustment"""
        rows = [cell for cell in cells if cell["label"] == "table row"]
        columns = [cell for cell in cells if cell["label"] == "table column"]

        # Sort rows and columns based on original orientation
        rows.sort(key=lambda x: x["bbox"][1])  # Sort by Y coordinate
        columns.sort(key=lambda x: x["bbox"][0])  # Sort by X coordinate

        cell_coordinates = []
        for row in rows:
            row_cells = []
            for column in columns:
                cell_bbox = [
                    column["bbox"][0],
                    row["bbox"][1],
                    column["bbox"][2],
                    row["bbox"][3]
                ]
                row_cells.append({
                    "column": column["bbox"],
                    "cell": cell_bbox
                })

            # Sort cells left-to-right based on original X coordinates
            row_cells.sort(key=lambda x: x["column"][0])
            cell_coordinates.append({
                "row": row["bbox"],
                "cells": row_cells,
                "cell_count": len(row_cells)
            })

        # Sort rows top-to-bottom based on original Y coordinates
        cell_coordinates.sort(key=lambda x: x["row"][1])
        return cell_coordinates

    def _process_ocr(self, cropped_table, cell_coords, page_num, page_dir):
        """Perform OCR with rotation awareness"""
        try:
            data = self.ocr.process_cells(cell_coords, cropped_table)
            csv_path = os.path.join(page_dir, f"page_{page_num}_data.csv")

            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for row in data.values():
                    # writer.writerow([page_num] + row)  # Add page number column
                    writer.writerow(row)  # Remove page column

            logger.info(f"Saved CSV for page {page_num} to {csv_path}")
            return csv_path

        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            return None


    def _merge_csvs(self, csv_paths):
        """Merge all page CSVs into a single file without page column"""
        if not csv_paths:
            return None
    
        merged_path = os.path.join(Config.OUTPUT_DIR, "merged_results.csv")
    
        with open(merged_path, "w", newline="", encoding="utf-8") as merged_file:
            writer = csv.writer(merged_file)
            header_written = False
    
            for csv_path in csv_paths:
                if not os.path.exists(csv_path):
                    continue
                
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    try:
                        header = next(reader)
                        if not header_written:
                            writer.writerow(header)  # Write original header
                            header_written = True
                    except StopIteration:
                        continue
                    
                    for row in reader:
                        writer.writerow(row)  # Write rows directly
    
        logger.info(f"Merged CSV created at {merged_path}")
        return merged_path
