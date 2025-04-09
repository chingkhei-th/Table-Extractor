"""Table detection model implementation."""

import torch
from transformers import AutoModelForObjectDetection
from src.utils import rescale_bboxes


class TableDetectionModel:
    """Model for detecting tables in documents."""

    def __init__(self, config, device="cpu"):
        """Initialize the table detection model.

        Args:
            config (TableExtractorConfig): Configuration object.
            device (str): Device to run model on ('cuda' or 'cpu').
        """
        self.config = config
        self.device = device

        # Load detection model
        print("Loading table detection model...")
        self.model = AutoModelForObjectDetection.from_pretrained(
            config.detection_model_name, revision=config.detection_model_revision
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

    def filter_objects(self, objects):
        """Filter objects based on confidence thresholds.

        Args:
            objects (list): List of detected objects.

        Returns:
            list: Filtered objects.
        """
        return [
            obj
            for obj in objects
            if obj["score"]
            >= self.config.detection_class_thresholds.get(obj["label"], 0.5)
        ]

    def objects_to_crops(self, img, objects, padding=None):
        """Process the bounding boxes to crop table images.

        Args:
            img (PIL.Image): Original image.
            objects (list): List of detected objects.
            padding (int, optional): Padding around crops. Defaults to config value.

        Returns:
            list: List of cropped table information.
        """
        if padding is None:
            padding = self.config.crop_padding

        table_crops = []

        for obj in objects:
            if obj["score"] < self.config.detection_class_thresholds.get(
                obj["label"], 0.5
            ):
                continue

            cropped_table = {}
            bbox = obj["bbox"]
            bbox = [
                max(0, bbox[0] - padding),
                max(0, bbox[1] - padding),
                min(img.width, bbox[2] + padding),
                min(img.height, bbox[3] + padding),
            ]

            cropped_img = img.crop(bbox)

            # If table is predicted to be rotated, just keep original orientation as requested
            cropped_table["image"] = cropped_img
            cropped_table["tokens"] = []  # Empty tokens list
            cropped_table["bbox"] = bbox
            cropped_table["label"] = obj["label"]

            table_crops.append(cropped_table)

        return table_crops
