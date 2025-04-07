# table_detector.py
import torch
from transformers import AutoModelForObjectDetection
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
from PIL import Image
import os
from src.utils import MaxResize, box_cxcywh_to_xyxy, rescale_bboxes, iob

# Add the missing import
try:
    from accelerate import init_empty_weights
except ImportError:
    # Fallback for older versions or if accelerate is not installed
    def init_empty_weights():
        """
        Dummy implementation to prevent the error.
        """
        class EmptyWeightsInitializer:
            def __enter__(self):
                return None
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        return EmptyWeightsInitializer()


def load_detection_model(device="cpu"):
    """
    Load the table detection model
    """
    try:
        model = AutoModelForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection", revision="no_timm"
        )
        model.to(device)
        return model
    except Exception as e:
        print(f"Error loading model with standard method: {e}")
        print("Trying alternative loading method...")
        
        # Alternative loading method if the standard one fails
        config = AutoModelForObjectDetection.config_class.from_pretrained(
            "microsoft/table-transformer-detection", revision="no_timm"
        )
        model = AutoModelForObjectDetection.from_config(config)
        
        # Load state dict manually
        state_dict = torch.hub.load_state_dict_from_url(
            "https://huggingface.co/microsoft/table-transformer-detection/resolve/main/pytorch_model.bin",
            map_location=device
        )
        model.load_state_dict(state_dict)
        model.to(device)
        return model


def detect_tables(image, threshold=0.5, visualize=False, output_dir=None):
    """
    Detect tables in an image

    Args:
        image (PIL.Image): Input image
        threshold (float): Detection threshold
        visualize (bool): Whether to save visualization
        output_dir (str): Directory to save visualizations

    Returns:
        list: List of cropped table images
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_detection_model(device)

    # Prepare image
    detection_transform = transforms.Compose(
        [
            MaxResize(800),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    pixel_values = detection_transform(image).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(pixel_values)

    # Process outputs
    id2label = model.config.id2label
    id2label[len(id2label)] = "no object"

    objects = outputs_to_objects(outputs, image.size, id2label)

    if visualize and output_dir:
        fig = visualize_detected_tables(image, objects)
        plt.savefig(
            os.path.join(output_dir, "detected_tables.jpg"),
            bbox_inches="tight",
            dpi=150,
        )
        plt.close()

    # Crop tables
    detection_class_thresholds = {
        "table": threshold,
        "table rotated": threshold,
        "no object": 10,
    }

    tables_crops = objects_to_crops(
        image, [], objects, detection_class_thresholds, padding=0
    )

    return [crop["image"].convert("RGB") for crop in tables_crops]


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


def visualize_detected_tables(img, det_tables):
    """
    Visualize detected tables in an image
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
    plt.axis("off")

    return plt.gcf()


def objects_to_crops(img, tokens, objects, class_thresholds, padding=10):
    """
    Process the bounding boxes into cropped table images and tokens
    """
    table_crops = []
    for obj in objects:
        if obj["score"] < class_thresholds[obj["label"]]:
            continue

        cropped_table = {}

        bbox = obj["bbox"]
        bbox = [
            bbox[0] - padding,
            bbox[1] - padding,
            bbox[2] + padding,
            bbox[3] + padding,
        ]

        cropped_img = img.crop(bbox)

        table_tokens = [token for token in tokens if iob(token["bbox"], bbox) >= 0.5]
        for token in table_tokens:
            token["bbox"] = [
                token["bbox"][0] - bbox[0],
                token["bbox"][1] - bbox[1],
                token["bbox"][2] - bbox[0],
                token["bbox"][3] - bbox[1],
            ]

        # If table is predicted to be rotated, rotate cropped image and tokens/words:
        if obj["label"] == "table rotated":
            cropped_img = cropped_img.rotate(270, expand=True)
            for token in table_tokens:
                bbox = token["bbox"]
                bbox = [
                    cropped_img.size[0] - bbox[3] - 1,
                    bbox[0],
                    cropped_img.size[0] - bbox[1] - 1,
                    bbox[2],
                ]
                token["bbox"] = bbox

        cropped_table["image"] = cropped_img
        cropped_table["tokens"] = table_tokens

        table_crops.append(cropped_table)

    return table_crops
