"""Template-based column detection for tables."""

import torch
from src.utils import get_transform


class TemplateColumnDetector:
    """Column detector using first page as a template."""

    def __init__(self, config, detection_model, structure_model, device="cpu"):
        """Initialize the template-based column detector.

        Args:
            config (TableExtractorConfig): Configuration object.
            detection_model (TableDetectionModel): Detection model.
            structure_model (TableStructureModel): Structure model.
            device (str): Device to run model on ('cuda' or 'cpu').
        """
        self.config = config
        self.detection_model = detection_model
        self.structure_model = structure_model
        self.device = device
        self.template_columns = None

        # Set up transforms
        self.detection_transform = get_transform(
            config.max_detection_size, is_detection=True
        )
        self.structure_transform = get_transform(
            config.max_structure_size, is_detection=False
        )

    def extract_column_template(self, image):
        """Extract column structure from the first page to use as a template.

        Args:
            image (PIL.Image): First page image.

        Returns:
            list: Normalized column positions, or None if no columns detected.
        """
        # Table detection
        pixel_values = self.detection_transform(image).unsqueeze(0).to(self.device)
        outputs = self.detection_model.predict(pixel_values)

        # Extract table objects
        objects = self.detection_model.outputs_to_objects(outputs, image.size)

        if not objects:
            print("No tables detected on first page for template")
            return None

        # Get the first table
        table_crops = self.detection_model.objects_to_crops(image, objects)
        if not table_crops:
            return None

        # Get the cropped table image
        cropped_table = table_crops[0]["image"]

        # Apply structure recognition
        pixel_values = (
            self.structure_transform(cropped_table).unsqueeze(0).to(self.device)
        )
        structure_outputs = self.structure_model.predict(pixel_values)

        # Extract cell structure
        cells = self.structure_model.outputs_to_objects(
            structure_outputs, cropped_table.size
        )

        # Extract just the columns and normalize their positions
        columns = [entry for entry in cells if entry["label"] == "table column"]

        if not columns:
            print("No columns detected in the first page table")
            return None

        # Sort columns by x-coordinate
        columns.sort(key=lambda x: x["bbox"][0])

        # Get table width
        table_width = cropped_table.width

        # Normalize column positions (as ratios of table width)
        normalized_columns = []
        for col in columns:
            normalized_col = {
                "x_start_ratio": col["bbox"][0] / table_width,
                "x_end_ratio": col["bbox"][2] / table_width,
                "original_bbox": col["bbox"],
            }
            normalized_columns.append(normalized_col)

        return normalized_columns

    def apply_template_to_table(self, cropped_table, cells):
        """Apply template columns to the detected rows.

        Args:
            cropped_table (PIL.Image): Cropped table image.
            cells (list): List of detected cells (rows, columns, etc.).

        Returns:
            list: Combined list of rows and template columns.
        """
        if not self.template_columns:
            return cells

        # Extract rows from detected cells
        rows = [entry for entry in cells if entry["label"] == "table row"]

        # Sort rows by Y coordinate
        rows.sort(key=lambda x: x["bbox"][1])

        # Table width
        table_width = cropped_table.width
        template_applied_cells = []

        # Add rows to cells
        for row in rows:
            template_applied_cells.append(row)

        # Create column objects from template
        for col in self.template_columns:
            x_start = col["x_start_ratio"] * table_width
            x_end = col["x_end_ratio"] * table_width

            # Create column object
            column_obj = {
                "label": "table column",
                "score": 0.99,  # High confidence since it's from template
                "bbox": [x_start, 0, x_end, cropped_table.height],
            }

            template_applied_cells.append(column_obj)

        return template_applied_cells
