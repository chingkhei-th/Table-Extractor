"""Utility functions for image processing."""

import io
import torch
import numpy as np
from PIL import Image
import pdf2image
from torchvision import transforms


class MaxResize:
    """Resizes an image to a maximum size while maintaining aspect ratio."""

    def __init__(self, max_size=800):
        """Initialize the resizer with a maximum size.

        Args:
            max_size (int): Maximum dimension (width or height) in pixels.
        """
        self.max_size = max_size

    def __call__(self, image):
        """Resize the image while maintaining aspect ratio.

        Args:
            image (PIL.Image): Input image to resize.

        Returns:
            PIL.Image: Resized image.
        """
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize(
            (int(round(scale * width)), int(round(scale * height)))
        )
        return resized_image


def convert_pdf_to_images(pdf_path, dpi=200):
    """Convert PDF file to a list of PIL images.

    Args:
        pdf_path (str): Path to the PDF file.
        dpi (int): Resolution in dots per inch.

    Returns:
        list: List of PIL Image objects.
    """
    print(f"Converting PDF to images: {pdf_path} (DPI: {dpi})")
    return pdf2image.convert_from_path(pdf_path, dpi=dpi)


def box_cxcywh_to_xyxy(x):
    """Convert bounding box from center format to corner format.

    Args:
        x (torch.Tensor): Bounding box in center format (cx, cy, w, h).

    Returns:
        torch.Tensor: Bounding box in corner format (x1, y1, x2, y2).
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    """Rescale bounding boxes according to image size.

    Args:
        out_bbox (torch.Tensor): Bounding boxes to rescale.
        size (tuple): Image size (width, height).

    Returns:
        torch.Tensor: Rescaled bounding boxes.
    """
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image.

    Args:
        fig (matplotlib.figure.Figure): Matplotlib figure object.

    Returns:
        PIL.Image: Image representation of the figure.
    """
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def iob(bbox1, bbox2):
    """Compute the intersection over box area ratio between two bounding boxes.
    Used to determine if a token belongs to a table.

    Args:
        bbox1 (list): First bounding box [x1, y1, x2, y2].
        bbox2 (list): Second bounding box [x1, y1, x2, y2].

    Returns:
        float: Intersection over box area ratio.
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


def get_transform(max_size, is_detection=True):
    """Create image transform pipeline based on configuration.

    Args:
        max_size (int): Maximum image dimension.
        is_detection (bool): Whether this is for detection or structure model.

    Returns:
        torchvision.transforms.Compose: Transform pipeline.
    """
    return transforms.Compose(
        [
            MaxResize(max_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
