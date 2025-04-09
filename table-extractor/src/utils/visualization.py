"""Visualization utilities for table extraction."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
from PIL import Image, ImageDraw

from src.utils.image_utils import fig2img


def visualize_detected_tables(img, det_tables):
    """Visualize detected tables on the image.

    Args:
        img (PIL.Image): Image to visualize on.
        det_tables (list): List of detected table dictionaries.

    Returns:
        PIL.Image: Visualization image.
    """
    plt.figure(figsize=(20, 20))
    plt.imshow(img, interpolation="lanczos")
    ax = plt.gca()

    for det_table in det_tables:
        bbox = det_table["bbox"]

        if det_table["label"] == "table":
            facecolor = (1, 0, 0.45)
            edgecolor = (1, 0, 0.45)
            alpha = 0.3
            linewidth = 2
            hatch = "//////"
        elif det_table["label"] == "table rotated":
            facecolor = (0.95, 0.6, 0.1)
            edgecolor = (0.95, 0.6, 0.1)
            alpha = 0.3
            linewidth = 2
            hatch = "//////"
        else:
            continue

        rect = patches.Rectangle(
            bbox[:2],
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=linewidth,
            edgecolor="none",
            facecolor=facecolor,
            alpha=0.1,
        )
        ax.add_patch(rect)
        rect = patches.Rectangle(
            bbox[:2],
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor="none",
            linestyle="-",
            alpha=alpha,
        )
        ax.add_patch(rect)
        rect = patches.Rectangle(
            bbox[:2],
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=0,
            edgecolor=edgecolor,
            facecolor="none",
            linestyle="-",
            hatch=hatch,
            alpha=0.2,
        )
        ax.add_patch(rect)

    plt.xticks([], [])
    plt.yticks([], [])

    legend_elements = [
        Patch(
            facecolor=(1, 0, 0.45),
            edgecolor=(1, 0, 0.45),
            label="Table",
            hatch="//////",
            alpha=0.3,
        ),
        Patch(
            facecolor=(0.95, 0.6, 0.1),
            edgecolor=(0.95, 0.6, 0.1),
            label="Table (rotated)",
            hatch="//////",
            alpha=0.3,
        ),
    ]

    plt.legend(
        handles=legend_elements,
        bbox_to_anchor=(0.5, -0.02),
        loc="upper center",
        borderaxespad=0,
        fontsize=10,
        ncol=2,
    )
    plt.gcf().set_size_inches(10, 10)
    plt.axis("off")

    fig = plt.gcf()
    visualized_image = fig2img(fig)
    plt.close()

    return visualized_image


def visualize_table_structure(cropped_table, cells):
    """Visualize the detected cells on the cropped table image.

    Args:
        cropped_table (PIL.Image): Cropped table image.
        cells (list): List of cell dictionaries.

    Returns:
        PIL.Image: Visualization with cell boundaries.
    """
    cropped_table_visualized = cropped_table.copy()
    draw = ImageDraw.Draw(cropped_table_visualized)

    for cell in cells:
        draw.rectangle(cell["bbox"], outline="red")

    return cropped_table_visualized
