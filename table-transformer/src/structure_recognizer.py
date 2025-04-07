# structure_recognizer.py
import torch
from transformers import TableTransformerForObjectDetection
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import ImageDraw, Image
import os
from src.utils import MaxResize, outputs_to_objects


def load_structure_model(device="cpu"):
    """
    Load the table structure recognition model
    """
    model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-structure-recognition-v1.1-all"
    )
    model.to(device)
    return model


def recognize_structure(table_image, visualize=False, output_dir=None, table_idx=0):
    """
    Recognize the structure of a table image

    Args:
        table_image (PIL.Image): Cropped table image
        visualize (bool): Whether to save visualization
        output_dir (str): Directory to save visualizations
        table_idx (int): Table index for naming visualization files

    Returns:
        tuple: (cells, cell_coordinates) Detected cells and their coordinates
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_structure_model(device)

    # Prepare image
    structure_transform = transforms.Compose(
        [
            MaxResize(1000),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    pixel_values = structure_transform(table_image).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(pixel_values)

    # Process outputs
    structure_id2label = model.config.id2label
    structure_id2label[len(structure_id2label)] = "no object"

    cells = outputs_to_objects(outputs, table_image.size, structure_id2label)

    if visualize and output_dir:
        # Visualize cells
        table_visualized = table_image.copy()
        draw = ImageDraw.Draw(table_visualized)

        for cell in cells:
            draw.rectangle(cell["bbox"], outline="red")

        table_visualized.save(
            os.path.join(output_dir, f"table_{table_idx+1}_structure.jpg")
        )

        # Visualize rows
        plt.figure(figsize=(16, 10))
        plt.imshow(table_image)
        ax = plt.gca()

        for cell in cells:
            if cell["label"] == "table row":
                bbox = cell["bbox"]
                ax.add_patch(
                    plt.Rectangle(
                        (bbox[0], bbox[1]),
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1],
                        fill=False,
                        color="red",
                        linewidth=3,
                    )
                )
                text = f'{cell["label"]}: {cell["score"]:0.2f}'
                ax.text(
                    bbox[0],
                    bbox[1],
                    text,
                    fontsize=15,
                    bbox=dict(facecolor="yellow", alpha=0.5),
                )

        plt.axis("off")
        plt.savefig(
            os.path.join(output_dir, f"table_{table_idx+1}_rows.jpg"),
            bbox_inches="tight",
        )
        plt.close()

    # Get cell coordinates by row
    cell_coordinates = get_cell_coordinates_by_row(cells)

    return cells, cell_coordinates


def get_cell_coordinates_by_row(table_data):
    """
    Extract cell coordinates organized by rows
    """
    # Extract rows and columns
    rows = [entry for entry in table_data if entry["label"] == "table row"]
    columns = [entry for entry in table_data if entry["label"] == "table column"]

    # Sort rows and columns by their Y and X coordinates, respectively
    rows.sort(key=lambda x: x["bbox"][1])
    columns.sort(key=lambda x: x["bbox"][0])

    # Find cell coordinates
    def find_cell_coordinates(row, column):
        cell_bbox = [
            column["bbox"][0],
            row["bbox"][1],
            column["bbox"][2],
            row["bbox"][3],
        ]
        return cell_bbox

    # Generate cell coordinates
    cell_coordinates = []

    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({"column": column["bbox"], "cell": cell_bbox})

        # Sort cells in the row by X coordinate
        row_cells.sort(key=lambda x: x["column"][0])

        # Append row information
        cell_coordinates.append(
            {"row": row["bbox"], "cells": row_cells, "cell_count": len(row_cells)}
        )

    # Sort rows from top to bottom
    cell_coordinates.sort(key=lambda x: x["row"][1])

    return cell_coordinates
