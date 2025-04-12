import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

class TableDetector:
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the TableDetector

        Args:
            config (Dict[str, Any], optional): Configuration dictionary
        """
        self.config = config or {}

    def detect_tables(self, img: np.ndarray, page_num: int) -> List[Tuple[int, int, int, int]]:
        """
        Detect tables in an image

        Args:
            img (np.ndarray): Image as numpy array
            page_num (int): Page number

        Returns:
            List[Tuple[int, int, int, int]]: List of table regions as (x, y, w, h)
        """
        return self._detect_table_regions(img, page_num)

    def _detect_table_regions(self, img: np.ndarray, page_num: int) -> List[Tuple[int, int, int, int]]:
        """
        Detect potential table regions in an image
        Args:
            img (np.ndarray): Image as numpy array
            page_num (int): Page number
        Returns:
            List[Tuple[int, int, int, int]]: List of table regions as (x, y, w, h)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply denoising if enabled
        if self.config.get('denoise', True):
            gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        # Apply morphological operations to enhance table boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        if self.config.get('morph_close', True):
            # Close operation to connect nearby lines
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        # Dilate to make lines more prominent
        dilate = cv2.dilate(thresh, kernel, iterations=2)
        # Find contours
        contours, _ = cv2.findContours(
            dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        table_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter out small contours (likely not tables)
            min_width = img.shape[1] * 0.2
            min_height = img.shape[0] * 0.05
            if w < min_width or h < min_height:
                continue
            # Check for overlapping with already detected regions
            is_overlapping = False
            for rx, ry, rw, rh in table_regions:
                # Calculate overlap
                x_overlap = max(0, min(x + w, rx + rw) - max(x, rx))
                y_overlap = max(0, min(y + h, ry + rh) - max(y, ry))
                overlap_area = x_overlap * y_overlap
                contour_area = w * h
                if overlap_area > contour_area * 0.5:  # 50% overlap threshold
                    is_overlapping = True
                    break
            if is_overlapping:
                continue
            # Add to table regions
            table_regions.append((x, y, w, h))
        # If no tables detected, try a fallback approach
        if not table_regions:
            print(f"No tables detected on page {page_num}.")
            
        return table_regions