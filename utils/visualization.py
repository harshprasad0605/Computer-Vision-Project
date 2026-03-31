"""
Visualization utilities for displaying and saving comparison images.
Uses matplotlib for grid layouts and side-by-side comparisons.
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for CLI usage
import matplotlib.pyplot as plt

from config import OUTPUT_DIR


def _bgr_to_rgb(image):
    """Convert BGR image to RGB for matplotlib display."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def show_comparison(original, processed, title_original="Original",
                    title_processed="Processed", save_name=None,
                    subdirectory=None):
    """
    Create a side-by-side comparison of two images and save to disk.

    Args:
        original (numpy.ndarray): Original image.
        processed (numpy.ndarray): Processed image.
        title_original (str): Title for the left image.
        title_processed (str): Title for the right image.
        save_name (str, optional): Filename to save the comparison.
        subdirectory (str, optional): Subdirectory within OUTPUT_DIR.

    Returns:
        str: Path to saved comparison image (if save_name provided).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Original
    orig_display = _bgr_to_rgb(original)
    if len(orig_display.shape) == 2:
        axes[0].imshow(orig_display, cmap="gray")
    else:
        axes[0].imshow(orig_display)
    axes[0].set_title(title_original, fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Processed
    proc_display = _bgr_to_rgb(processed)
    if len(proc_display.shape) == 2:
        axes[1].imshow(proc_display, cmap="gray")
    else:
        axes[1].imshow(proc_display)
    axes[1].set_title(title_processed, fontsize=14, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()

    if save_name:
        out_dir = os.path.join(OUTPUT_DIR, subdirectory) if subdirectory else OUTPUT_DIR
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, save_name)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[SAVED] {out_path}")
        plt.close(fig)
        return out_path

    plt.close(fig)
    return None


def show_image_grid(images, titles, grid_cols=3, save_name=None,
                    subdirectory=None, main_title=None):
    """
    Display a grid of images with titles.

    Args:
        images (list[numpy.ndarray]): List of images.
        titles (list[str]): List of titles for each image.
        grid_cols (int): Number of columns in the grid.
        save_name (str, optional): Filename to save.
        subdirectory (str, optional): Subdirectory within OUTPUT_DIR.
        main_title (str, optional): Overall figure title.

    Returns:
        str: Path to saved image (if save_name provided).
    """
    n = len(images)
    grid_rows = (n + grid_cols - 1) // grid_cols
    fig, axes = plt.subplots(grid_rows, grid_cols,
                              figsize=(5 * grid_cols, 5 * grid_rows))

    if grid_rows == 1 and grid_cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx in range(grid_rows * grid_cols):
        row, col = divmod(idx, grid_cols)
        ax = axes[row][col]
        if idx < n:
            img = _bgr_to_rgb(images[idx])
            if len(img.shape) == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)
            ax.set_title(titles[idx], fontsize=12, fontweight="bold")
        ax.axis("off")

    if main_title:
        fig.suptitle(main_title, fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_name:
        out_dir = os.path.join(OUTPUT_DIR, subdirectory) if subdirectory else OUTPUT_DIR
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, save_name)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[SAVED] {out_path}")
        plt.close(fig)
        return out_path

    plt.close(fig)
    return None
