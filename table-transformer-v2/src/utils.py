import io
import os
import csv
import glob
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf)

def iob(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    return intersection / area1 if area1 > 0 else 0

class MaxResize:
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        scale = self.max_size / max(width, height)
        return image.resize((int(round(width*scale)), int(round(height*scale))))