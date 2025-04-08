from transformers import AutoModelForObjectDetection
from torchvision import transforms
from src.utils import MaxResize
from src.config import Config
import torch

class TableDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForObjectDetection.from_pretrained(
            Config.DETECTION_MODEL_NAME,
            revision="no_timm"
        ).to(self.device)
        self.transform = self._create_transform()

    def _create_transform(self):
        return transforms.Compose([
            MaxResize(800),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def detect(self, image):
        pixel_values = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(pixel_values)
        return outputs