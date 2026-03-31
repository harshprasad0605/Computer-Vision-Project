"""
Image Segmentation Module
==========================
Implements region-based and boundary-based segmentation methods:
- Global thresholding (Otsu's method)
- Adaptive thresholding (mean and Gaussian weighted)
- Watershed segmentation (marker-based)
- GrabCut segmentation (interactive foreground extraction)

Segmentation partitions an image into meaningful regions, enabling
tasks like object extraction, scene understanding, and analysis.
"""

import cv2
import numpy as np

from config import (
    ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C,
    GRABCUT_ITER_COUNT, GRABCUT_RECT_MARGIN,
)
from utils.io_utils import ensure_grayscale, ensure_color
from utils.visualization import show_image_grid, show_comparison


def threshold_otsu(image):
    """
    Apply Otsu's automatic thresholding.

    Otsu's method finds the optimal threshold that minimizes the
    intra-class variance (or equivalently maximizes inter-class variance)
    of the two pixel classes (foreground and background). It works best
    for images with bimodal histograms.

    Args:
        image: Input image.

    Returns:
        tuple: (threshold_value, binary_image)
    """
    gray = ensure_grayscale(image)
    thresh_val, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"  Otsu threshold value: {thresh_val:.1f}")
    return thresh_val, binary


def threshold_adaptive(image, method="gaussian", block_size=None, C=None):
    """
    Apply adaptive thresholding for images with uneven illumination.

    Unlike global thresholding, adaptive thresholding computes a
    separate threshold for each pixel based on its local neighbourhood,
    making it effective for images with varying lighting conditions.

    Args:
        image: Input image.
        method: 'mean' or 'gaussian'. Gaussian gives more weight to
                pixels closer to the centre of the neighbourhood.
        block_size: Size of the local neighbourhood (must be odd).
        C: Constant subtracted from the computed threshold.

    Returns:
        Binary image.
    """
    block_size = block_size or ADAPTIVE_BLOCK_SIZE
    C = C or ADAPTIVE_C
    gray = ensure_grayscale(image)

    adapt_method = (
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == "gaussian"
        else cv2.ADAPTIVE_THRESH_MEAN_C
    )
    return cv2.adaptiveThreshold(
        gray, 255, adapt_method, cv2.THRESH_BINARY, block_size, C
    )


def watershed_segmentation(image):
    """
    Apply marker-based watershed segmentation.

    The watershed algorithm treats the image as a topographic surface
    where pixel intensity represents elevation. It "floods" from
    markers placed in sure foreground and background regions, and
    builds dams at the meeting points — these dams are the segmentation
    boundaries.

    Steps:
    1. Convert to grayscale and threshold
    2. Morphological opening to remove noise
    3. Dilate to find sure background
    4. Distance transform to find sure foreground
    5. Identify unknown region
    6. Apply watershed with markers

    Args:
        image: Input BGR image.

    Returns:
        tuple: (segmented_image, markers)
            - segmented_image: Image with segment boundaries marked in red
            - markers: Integer label map of segments
    """
    gray = ensure_grayscale(image)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological opening to remove noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background via dilation
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Sure foreground via distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # background is 1, not 0
    markers[unknown == 255] = 0  # unknown regions are 0

    # Apply watershed
    result = image.copy()
    markers = cv2.watershed(result, markers)
    result[markers == -1] = [0, 0, 255]  # Mark boundaries in red

    n_segments = len(np.unique(markers)) - 1  # exclude boundary marker
    print(f"  Watershed found {n_segments} segments")

    return result, markers


def grabcut_segmentation(image, rect=None, iter_count=None):
    """
    Apply GrabCut for foreground extraction.

    GrabCut uses an iterative energy minimization approach based on
    Gaussian Mixture Models. Given an initial rectangle around the
    foreground object, it iteratively refines the foreground/background
    classification using graph cuts.

    Args:
        image: Input BGR image.
        rect: Bounding rectangle (x, y, w, h) for initial foreground.
              If None, uses a margin from the image borders.
        iter_count: Number of iterations.

    Returns:
        tuple: (foreground_image, mask)
            - foreground_image: Image with background removed
            - mask: Segmentation mask
    """
    iter_count = iter_count or GRABCUT_ITER_COUNT
    h, w = image.shape[:2]
    margin = GRABCUT_RECT_MARGIN

    if rect is None:
        rect = (margin, margin, w - 2 * margin, h - 2 * margin)

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(image, mask, rect, bgd_model, fgd_model,
                iter_count, cv2.GC_INIT_WITH_RECT)

    # Create binary mask: foreground = definite fg + probable fg
    mask_binary = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype("uint8")

    foreground = cv2.bitwise_and(image, image, mask=mask_binary)
    return foreground, mask_binary


def compare_all_segmentation(image, save_name="segmentation_comparison.png"):
    """
    Apply all segmentation methods and create a comparison grid.

    Args:
        image: Input BGR image.
        save_name: Output filename.

    Returns:
        str: Path to the saved comparison image.
    """
    gray = ensure_grayscale(image)
    _, otsu = threshold_otsu(image)
    adaptive_gauss = threshold_adaptive(image, method="gaussian")
    adaptive_mean = threshold_adaptive(image, method="mean")
    watershed_result, _ = watershed_segmentation(image)
    grabcut_result, _ = grabcut_segmentation(image)

    images = [
        gray, otsu, adaptive_gauss,
        adaptive_mean, watershed_result, grabcut_result,
    ]
    titles = [
        "Original (Gray)", "Otsu Thresholding", "Adaptive (Gaussian)",
        "Adaptive (Mean)", "Watershed", "GrabCut",
    ]
    return show_image_grid(
        images, titles, grid_cols=3,
        save_name=save_name, subdirectory="segmentation",
        main_title="Image Segmentation Comparison"
    )
