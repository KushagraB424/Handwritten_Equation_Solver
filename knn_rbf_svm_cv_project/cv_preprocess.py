import cv2
import numpy as np
from typing import Tuple


def to_gray_and_binarize(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold (Gaussian) for robustness to illumination
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    return gray, bin_img


def denoise_and_normalize(bin_img: np.ndarray) -> np.ndarray:
    # Remove salt-and-pepper noise
    den = cv2.medianBlur(bin_img, 3)
    # Morphological open to remove small specks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    den = cv2.morphologyEx(den, cv2.MORPH_OPEN, kernel, iterations=1)
    return den


def _deskew_moments(gray: np.ndarray) -> Tuple[np.ndarray, float]:
    # Compute skew using image moments on binary text mask
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(th > 0))
    if coords.size == 0:
        return gray, 0.0
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


def deskew(gray_or_bin: np.ndarray) -> Tuple[np.ndarray, float]:
    if len(gray_or_bin.shape) == 3:
        gray = cv2.cvtColor(gray_or_bin, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_or_bin.copy()
    rotated, angle = _deskew_moments(gray)
    return rotated, angle


def preprocess_image(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    gray, bin_img = to_gray_and_binarize(img)
    den = denoise_and_normalize(bin_img)
    deskewed, angle = deskew(gray)
    # Recompute binary after deskew to keep alignment
    _, bin_deskew = cv2.threshold(deskewed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return {
        'original': img,
        'gray': gray,
        'binary': bin_deskew,
        'deskew_angle': angle
    }