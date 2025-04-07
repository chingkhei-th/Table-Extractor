# utils.py
import torch
import csv
import os
from PIL import Image


class MaxResize(object):
    """
    Resize image to a maximum size while maintaining aspect ratio
    """

    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize(
            (int(round(scale * width)), int(round(scale * height)))
        )
        return resized_image


def box_cxcywh_to_xyxy(x):
    """
    Convert bounding box from (center_x, center_y, width, height) to (x1, y1, x2, y2)
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    """
    Rescale bounding boxes to match image size
    """
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def iob(box1, box2):
    """
    Compute the intersection over box area ratio between two boxes
    Used for token-table assignment
    """
    # Intersection box
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Box1 area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])

    # IoB
    if box1_area > 0:
        return intersection / box1_area
    else:
        return 0


def outputs_to_objects(outputs, img_size, id2label):
    """
    Convert model outputs to list of detected objects
    """
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == "no object":
            objects.append(
                {
                    "label": class_label,
                    "score": float(score),
                    "bbox": [float(elem) for elem in bbox],
                }
            )

    return objects


def save_to_csv(data, output_path):
    """
    Save table data to CSV
    """
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for _, row_data in data.items():
            writer.writerow(row_data)


def merge_csv_files(all_data, output_path):
    """
    Merge multiple table data into a single CSV
    """
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Add a separator row between tables
        separator = [""] * 5

        first_table = True
        for table_data in all_data:
            if not first_table:
                writer.writerow(separator)

            for _, row_data in table_data.items():
                writer.writerow(row_data)

            first_table = False
