"""
Edge Detection Module
=====================
Implements gradient-based and Laplacian edge detection methods:
- Sobel operator (horizontal, vertical, combined)
- Canny edge detector
- Laplacian of Gaussian

Each method highlights discontinuities (edges) in image intensity,
which correspond to object boundaries, texture changes, or shadows.
"""

import cv2
import numpy as np

from config import (
    CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD,
    SOBEL_KERNEL_SIZE, LAPLACIAN_KERNEL_SIZE,
)
from utils.io_utils import ensure_grayscale
from utils.visualization import show_image_grid, show_comparison


def detect_edges_sobel(image, ksize=None):
    """
    Detect edges using the Sobel operator.

    The Sobel operator computes the gradient of image intensity using
    two 3×3 kernels (horizontal and vertical). The gradient magnitude
    approximates the rate of intensity change — high values indicate edges.

    Args:
        image: Input image (BGR or grayscale).
        ksize: Size of the Sobel kernel (1, 3, 5, or 7).

    Returns:
        dict with keys: 'sobel_x', 'sobel_y', 'magnitude'
            - sobel_x: Horizontal gradient
            - sobel_y: Vertical gradient
            - magnitude: Combined gradient magnitude
    """
    ksize = ksize or SOBEL_KERNEL_SIZE
    gray = ensure_grayscale(image)

    # Compute gradients in x and y directions
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    # Compute gradient magnitude
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    magnitude = np.uint8(np.clip(magnitude / magnitude.max() * 255, 0, 255))

    sobel_x_display = np.uint8(np.clip(np.abs(sobel_x) / np.abs(sobel_x).max() * 255, 0, 255))
    sobel_y_display = np.uint8(np.clip(np.abs(sobel_y) / np.abs(sobel_y).max() * 255, 0, 255))

    return {
        "sobel_x": sobel_x_display,
        "sobel_y": sobel_y_display,
        "magnitude": magnitude,
    }


def detect_edges_canny(image, low_thresh=None, high_thresh=None):
    """
    Detect edges using the Canny edge detector.

    Canny edge detection is a multi-stage algorithm:
    1. Gaussian smoothing to reduce noise
    2. Gradient computation using Sobel operators
    3. Non-maximum suppression to thin edges
    4. Double thresholding and edge tracking by hysteresis

    Args:
        image: Input image.
        low_thresh: Lower hysteresis threshold.
        high_thresh: Upper hysteresis threshold.

    Returns:
        Binary edge map (uint8).
    """
    low_thresh = low_thresh or CANNY_LOW_THRESHOLD
    high_thresh = high_thresh or CANNY_HIGH_THRESHOLD
    gray = ensure_grayscale(image)
    return cv2.Canny(gray, low_thresh, high_thresh)


def detect_edges_laplacian(image, ksize=None):
    """
    Detect edges using the Laplacian operator.

    The Laplacian computes the second derivative of the image intensity.
    Zero-crossings in the Laplacian correspond to edges. It detects edges
    in all directions simultaneously but is more sensitive to noise.

    Args:
        image: Input image.
        ksize: Size of the Laplacian kernel.

    Returns:
        Edge-detected image (uint8).
    """
    ksize = ksize or LAPLACIAN_KERNEL_SIZE
    gray = ensure_grayscale(image)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    return np.uint8(np.clip(np.abs(laplacian), 0, 255))


def compare_all_edges(image, save_name="edge_comparison.png"):
    """
    Apply all edge detection methods and create a comparison grid.

    Args:
        image: Input image.
        save_name: Output filename.

    Returns:
        str: Path to the saved comparison image.
    """
    gray = ensure_grayscale(image)
    sobel_result = detect_edges_sobel(image)
    canny_result = detect_edges_canny(image)
    laplacian_result = detect_edges_laplacian(image)

    images = [
        gray,
        sobel_result["sobel_x"],
        sobel_result["sobel_y"],
        sobel_result["magnitude"],
        canny_result,
        laplacian_result,
    ]
    titles = [
        "Original (Grayscale)",
        "Sobel X (Horizontal)",
        "Sobel Y (Vertical)",
        "Sobel Magnitude",
        "Canny Edges",
        "Laplacian Edges",
    ]
    return show_image_grid(
        images, titles, grid_cols=3,
        save_name=save_name, subdirectory="edges",
        main_title="Edge Detection Comparison"
    )
