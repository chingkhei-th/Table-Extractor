from transformers import TableTransformerForObjectDetection
from torchvision import transforms
from src.utils import MaxResize
from src.config import Config
import torch

class TableRecognizer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TableTransformerForObjectDetection.from_pretrained(
            Config.STRUCTURE_MODEL_NAME
        ).to(self.device)
        self.transform = self._create_transform()

    def _create_transform(self):
        return transforms.Compose([
            MaxResize(1000),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def recognize(self, image):
        pixel_values = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(pixel_values)
        return outputs