"""Table structure recognition model implementation."""

import torch
from transformers import TableTransformerForObjectDetection
from src.utils import rescale_bboxes


class TableStructureModel:
    """Model for recognizing table structure (rows, columns, cells)."""

    def __init__(self, config, device="cpu"):
        """Initialize the table structure model.

        Args:
            config (TableExtractorConfig): Configuration object.
            device (str): Device to run model on ('cuda' or 'cpu').
        """
        self.config = config
        self.device = device

        # Load structure model
        print("Loading table structure recognition model...")
        self.model = TableTransformerForObjectDetection.from_pretrained(
            config.structure_model_name
        ).to(device)

        # Setup id2label with "no object" category
        self.id2label = self.model.config.id2label
        self.id2label[len(self.id2label)] = "no object"

    def predict(self, pixel_values):
        """Run inference on an image.

        Args:
            pixel_values (torch.Tensor): Preprocessed image tensor.

        Returns:
            dict: Model outputs.
        """
        with torch.no_grad():
            outputs = self.model(pixel_values)
        return outputs

    def outputs_to_objects(self, outputs, img_size):
        """Convert model outputs to object list with labels, scores, and bounding boxes.

        Args:
            outputs (dict): Model output dictionary.
            img_size (tuple): Image size (width, height).

        Returns:
            list: List of detected objects.
        """
        m = outputs.logits.softmax(-1).max(-1)
        pred_labels = list(m.indices.detach().cpu().numpy())[0]
        pred_scores = list(m.values.detach().cpu().numpy())[0]
        pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]
        pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

        objects = []
        for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
            class_label = self.id2label[int(label)]
            if not class_label == "no object":
                objects.append(
                    {
                        "label": class_label,
                        "score": float(score),
                        "bbox": [float(elem) for elem in bbox],
                    }
                )

        return objects

    def get_cell_coordinates_by_row(self, table_data):
        """Extract row and column information to create cell coordinates.

        Args:
            table_data (list): List of detected table elements.

        Returns:
            list: List of row dictionaries with cell information.
        """
        # Extract rows and columns
        rows = [entry for entry in table_data if entry["label"] == "table row"]
        columns = [entry for entry in table_data if entry["label"] == "table column"]

        # Sort rows and columns by their Y and X coordinates
        rows.sort(key=lambda x: x["bbox"][1])
        columns.sort(key=lambda x: x["bbox"][0])

        # Generate cell coordinates and count cells in each row
        cell_coordinates = []

        for row in rows:
            row_cells = []
            for column in columns:
                cell_bbox = [
                    column["bbox"][0],
                    row["bbox"][1],
                    column["bbox"][2],
                    row["bbox"][3],
                ]
                row_cells.append({"column": column["bbox"], "cell": cell_bbox})

            # Sort cells in the row by X coordinate
            row_cells.sort(key=lambda x: x["column"][0])

            # Append row information to cell_coordinates
            cell_coordinates.append(
                {"row": row["bbox"], "cells": row_cells, "cell_count": len(row_cells)}
            )

        # Sort rows from top to bottom
        cell_coordinates.sort(key=lambda x: x["row"][1])

        return cell_coordinates
