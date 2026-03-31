"""
I/O Utilities for image loading, validation, and saving.
Provides consistent error handling and path management.
"""

import os
import sys
import cv2
import numpy as np

from config import OUTPUT_DIR


def load_image(path, mode="color"):
    """
    Load an image from disk with validation.

    Args:
        path (str): Path to the image file.
        mode (str): 'color' for BGR, 'gray' for grayscale, 'unchanged' for original.

    Returns:
        numpy.ndarray: Loaded image.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image cannot be decoded.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")

    flags = {
        "color": cv2.IMREAD_COLOR,
        "gray": cv2.IMREAD_GRAYSCALE,
        "unchanged": cv2.IMREAD_UNCHANGED,
    }
    flag = flags.get(mode, cv2.IMREAD_COLOR)
    image = cv2.imread(path, flag)

    if image is None:
        raise ValueError(f"Failed to decode image: {path}")

    return image


def save_image(image, filename, subdirectory=None):
    """
    Save an image to the output directory.

    Args:
        image (numpy.ndarray): Image to save.
        filename (str): Output filename (e.g., 'result.jpg').
        subdirectory (str, optional): Subdirectory within OUTPUT_DIR.

    Returns:
        str: Full path to the saved image.
    """
    if subdirectory:
        out_dir = os.path.join(OUTPUT_DIR, subdirectory)
    else:
        out_dir = OUTPUT_DIR

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    cv2.imwrite(out_path, image)
    print(f"[SAVED] {out_path}")
    return out_path


def load_multiple_images(paths, mode="color"):
    """
    Load multiple images from a list of paths.

    Args:
        paths (list[str]): List of image file paths.
        mode (str): Color mode for all images.

    Returns:
        list[numpy.ndarray]: List of loaded images.
    """
    images = []
    for p in paths:
        images.append(load_image(p, mode))
    return images


def get_image_info(image):
    """
    Return a dictionary of basic image properties.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        dict: Dictionary with height, width, channels, dtype, size info.
    """
    h, w = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    return {
        "height": h,
        "width": w,
        "channels": channels,
        "dtype": str(image.dtype),
        "size_bytes": image.nbytes,
        "size_mb": round(image.nbytes / (1024 * 1024), 2),
    }


def print_image_info(image, name="Image"):
    """Print formatted image information to console."""
    info = get_image_info(image)
    print(f"\n{'='*40}")
    print(f"  {name} Information")
    print(f"{'='*40}")
    print(f"  Dimensions : {info['width']} x {info['height']}")
    print(f"  Channels   : {info['channels']}")
    print(f"  Dtype      : {info['dtype']}")
    print(f"  Size       : {info['size_mb']} MB")
    print(f"{'='*40}\n")


def ensure_grayscale(image):
    """Convert image to grayscale if it is not already."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def ensure_color(image):
    """Convert image to BGR color if it is grayscale."""
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image
