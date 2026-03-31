"""
Image Filtering Module
======================
Implements spatial domain filtering operations including:
- Gaussian blur
- Median filter
- Bilateral filter
- Custom convolution kernels (sharpening, embossing, etc.)

Each function accepts an image and returns the filtered result.
A comparison function generates a grid of all filter outputs.
"""

import cv2
import numpy as np

from config import (
    GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA,
    MEDIAN_KERNEL_SIZE,
    BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE,
)
from utils.visualization import show_image_grid, show_comparison


# ─────────────────────────────────────────────
# Predefined Custom Kernels
# ─────────────────────────────────────────────
CUSTOM_KERNELS = {
    "sharpen": np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32),

    "emboss": np.array([
        [-2, -1, 0],
        [-1,  1, 1],
        [ 0,  1, 2]
    ], dtype=np.float32),

    "box_blur": np.ones((5, 5), dtype=np.float32) / 25.0,

    "edge_enhance": np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ], dtype=np.float32),

    "ridge": np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float32),
}


def apply_gaussian_blur(image, ksize=None, sigma=None):
    """
    Apply Gaussian blur to smooth an image.

    The Gaussian filter convolves the image with a 2D Gaussian kernel,
    effectively averaging pixels weighted by their distance from the center.
    This reduces high-frequency noise while preserving overall structure.

    Args:
        image: Input image (BGR or grayscale).
        ksize: Kernel size as (width, height). Must be odd.
        sigma: Standard deviation of the Gaussian kernel.

    Returns:
        Blurred image.
    """
    ksize = ksize or GAUSSIAN_KERNEL_SIZE
    sigma = sigma or GAUSSIAN_SIGMA
    return cv2.GaussianBlur(image, ksize, sigma)


def apply_median_filter(image, ksize=None):
    """
    Apply median filter for salt-and-pepper noise removal.

    The median filter replaces each pixel with the median value in its
    neighbourhood. It is particularly effective at removing impulse noise
    while preserving edges better than linear filters.

    Args:
        image: Input image.
        ksize: Kernel size (must be odd positive integer).

    Returns:
        Filtered image.
    """
    ksize = ksize or MEDIAN_KERNEL_SIZE
    return cv2.medianBlur(image, ksize)


def apply_bilateral_filter(image, d=None, sigma_color=None, sigma_space=None):
    """
    Apply bilateral filter for edge-preserving smoothing.

    The bilateral filter considers both spatial proximity and intensity
    similarity, allowing it to smooth flat regions while preserving
    sharp edges. This makes it ideal for noise reduction in photographs.

    Args:
        image: Input image.
        d: Diameter of each pixel neighbourhood.
        sigma_color: Filter sigma in the color space.
        sigma_space: Filter sigma in the coordinate space.

    Returns:
        Filtered image.
    """
    d = d or BILATERAL_D
    sigma_color = sigma_color or BILATERAL_SIGMA_COLOR
    sigma_space = sigma_space or BILATERAL_SIGMA_SPACE
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def apply_custom_kernel(image, kernel_name="sharpen"):
    """
    Apply a predefined custom convolution kernel.

    Convolution slides a kernel matrix over the image, computing the
    weighted sum at each position. Different kernels produce different
    effects: sharpening emphasizes edges, embossing creates 3D relief,
    and ridge detection highlights ridges in the image.

    Args:
        image: Input image.
        kernel_name: One of 'sharpen', 'emboss', 'box_blur',
                     'edge_enhance', 'ridge'.

    Returns:
        Filtered image.

    Raises:
        ValueError: If kernel_name is not recognized.
    """
    if kernel_name not in CUSTOM_KERNELS:
        available = ", ".join(CUSTOM_KERNELS.keys())
        raise ValueError(
            f"Unknown kernel '{kernel_name}'. Available: {available}"
        )
    kernel = CUSTOM_KERNELS[kernel_name]
    return cv2.filter2D(image, -1, kernel)


def compare_all_filters(image, save_name="filter_comparison.png"):
    """
    Apply all filters to an image and produce a comparison grid.

    Args:
        image: Input image.
        save_name: Filename for the saved comparison.

    Returns:
        str: Path to the saved comparison image.
    """
    results = [
        image,
        apply_gaussian_blur(image),
        apply_median_filter(image),
        apply_bilateral_filter(image),
        apply_custom_kernel(image, "sharpen"),
        apply_custom_kernel(image, "emboss"),
    ]
    titles = [
        "Original",
        "Gaussian Blur",
        "Median Filter",
        "Bilateral Filter",
        "Sharpen Kernel",
        "Emboss Kernel",
    ]
    return show_image_grid(
        results, titles, grid_cols=3,
        save_name=save_name, subdirectory="filtering",
        main_title="Image Filtering Comparison"
    )
