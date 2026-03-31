"""
Feature Detection & Matching Module
====================================
Implements keypoint detection and descriptor matching methods:
- Harris corner detection
- ORB (Oriented FAST and Rotated BRIEF) feature detection
- Brute-force feature matching with ratio test

Feature detection identifies distinctive points in an image that are
invariant to certain transformations (scale, rotation), enabling
applications like image matching, object recognition, and tracking.
"""

import cv2
import numpy as np

from config import (
    HARRIS_BLOCK_SIZE, HARRIS_KSIZE, HARRIS_K, HARRIS_THRESHOLD,
    ORB_N_FEATURES, FEATURE_MATCH_RATIO,
)
from utils.io_utils import ensure_grayscale, ensure_color
from utils.visualization import show_image_grid, show_comparison


def detect_harris_corners(image, block_size=None, ksize=None, k=None,
                           threshold=None):
    """
    Detect corners using the Harris corner detector.

    Harris corner detection uses the structure tensor (second moment matrix)
    to measure the variation of intensity in a local window. Corners are
    points where the intensity changes significantly in both directions.

    The corner response function is: R = det(M) - k * (trace(M))^2
    where M is the structure tensor. High R values indicate corners.

    Args:
        image: Input image.
        block_size: Neighbourhood size for the structure tensor.
        ksize: Aperture parameter for the Sobel derivative.
        k: Harris detector sensitivity parameter.
        threshold: Fraction of max response above which points are corners.

    Returns:
        tuple: (corner_image, n_corners)
            - corner_image: Image with corners marked in red
            - n_corners: Number of detected corners
    """
    block_size = block_size or HARRIS_BLOCK_SIZE
    ksize = ksize or HARRIS_KSIZE
    k = k or HARRIS_K
    threshold = threshold or HARRIS_THRESHOLD

    gray = ensure_grayscale(image)
    result = ensure_color(image.copy()) if len(image.shape) == 2 else image.copy()

    # Compute Harris response
    harris = cv2.cornerHarris(gray, block_size, ksize, k)
    harris = cv2.dilate(harris, None)  # dilate for visibility

    # Mark corners
    corner_mask = harris > threshold * harris.max()
    result[corner_mask] = [0, 0, 255]  # Red

    n_corners = int(np.sum(corner_mask))
    print(f"  Harris corners detected: {n_corners}")
    return result, n_corners


def detect_orb_features(image, n_features=None):
    """
    Detect keypoints and compute descriptors using ORB.

    ORB (Oriented FAST and Rotated BRIEF) is a fast, efficient
    alternative to SIFT and SURF. It combines the FAST keypoint
    detector with the BRIEF descriptor, adding orientation
    computation for rotation invariance.

    Args:
        image: Input image.
        n_features: Maximum number of features to detect.

    Returns:
        tuple: (keypoint_image, keypoints, descriptors)
            - keypoint_image: Image with keypoints drawn
            - keypoints: List of cv2.KeyPoint objects
            - descriptors: Feature descriptor array
    """
    n_features = n_features or ORB_N_FEATURES
    gray = ensure_grayscale(image)

    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Draw keypoints with size and orientation
    result = cv2.drawKeypoints(
        image, keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    print(f"  ORB features detected: {len(keypoints)}")
    return result, keypoints, descriptors


def match_features(image1, image2, ratio=None):
    """
    Match features between two images using ORB + brute-force matching.

    Uses Lowe's ratio test to filter out poor matches:
    A match is kept only if the distance to the best match is
    significantly less than the distance to the second-best match.

    Args:
        image1: First input image.
        image2: Second input image.
        ratio: Lowe's ratio threshold (lower = stricter).

    Returns:
        tuple: (match_image, good_matches, kp1, kp2)
            - match_image: Side-by-side image with match lines drawn
            - good_matches: List of good DMatch objects
            - kp1, kp2: Keypoints for image1 and image2
    """
    ratio = ratio or FEATURE_MATCH_RATIO

    gray1 = ensure_grayscale(image1)
    gray2 = ensure_grayscale(image2)

    orb = cv2.ORB_create(nfeatures=ORB_N_FEATURES)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        print("  Warning: No descriptors found in one or both images.")
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        h = max(h1, h2)
        vis = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
        return vis, [], kp1, kp2

    # Brute-force matching with Hamming distance (for binary descriptors)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good_matches.append(m)

    # Draw matches
    match_image = cv2.drawMatches(
        image1, kp1, image2, kp2, good_matches, None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
    )

    print(f"  Total matches: {len(matches)}, Good matches: {len(good_matches)}")
    return match_image, good_matches, kp1, kp2


def compare_feature_detection(image, save_name="feature_comparison.png"):
    """
    Compare Harris and ORB detection on the same image.

    Args:
        image: Input image.
        save_name: Output filename.

    Returns:
        str: Path to saved comparison.
    """
    harris_img, _ = detect_harris_corners(image)
    orb_img, _, _ = detect_orb_features(image)

    images = [image, harris_img, orb_img]
    titles = ["Original", "Harris Corners", "ORB Features"]

    return show_image_grid(
        images, titles, grid_cols=3,
        save_name=save_name, subdirectory="features",
        main_title="Feature Detection Comparison"
    )
