"""
Image Stitching / Panorama Module
==================================
Creates panoramic images by stitching multiple overlapping images:
1. Feature detection (ORB)
2. Feature matching (brute-force with ratio test)
3. Homography estimation (RANSAC)
4. Perspective warping and blending

This module demonstrates projective geometry concepts including
homogeneous coordinates, homography matrices, and RANSAC-based
robust estimation.
"""

import cv2
import numpy as np

from config import (
    ORB_N_FEATURES, FEATURE_MATCH_RATIO,
    STITCH_REPROJECTION_THRESH, STITCH_MIN_MATCH_COUNT,
)
from utils.io_utils import ensure_grayscale
from utils.visualization import show_comparison


def stitch_images(image1, image2, n_features=None, ratio=None,
                  reproj_thresh=None, min_matches=None):
    """
    Stitch two overlapping images into a panorama.

    Pipeline:
    1. Detect ORB keypoints and compute descriptors
    2. Match descriptors using brute-force with Lowe's ratio test
    3. Estimate homography matrix using RANSAC
    4. Warp the second image to align with the first
    5. Blend the images together

    The homography H is a 3×3 matrix that maps points from one image
    plane to another: x' = H * x (in homogeneous coordinates).

    RANSAC (Random Sample Consensus) robustly estimates the homography
    by iteratively fitting models to random subsets and selecting the
    one with the most inliers.

    Args:
        image1: First (left) image.
        image2: Second (right) image.
        n_features: Number of ORB features to detect.
        ratio: Lowe's ratio test threshold.
        reproj_thresh: RANSAC reprojection threshold in pixels.
        min_matches: Minimum number of good matches required.

    Returns:
        tuple: (panorama, match_visualization, status_dict)
            - panorama: Stitched result image (or None on failure)
            - match_visualization: Image showing feature matches
            - status_dict: Dictionary with stitching statistics
    """
    n_features = n_features or ORB_N_FEATURES
    ratio = ratio or FEATURE_MATCH_RATIO
    reproj_thresh = reproj_thresh or STITCH_REPROJECTION_THRESH
    min_matches = min_matches or STITCH_MIN_MATCH_COUNT

    status = {"success": False, "total_matches": 0, "good_matches": 0,
              "inliers": 0}

    # Step 1: Detect features
    gray1 = ensure_grayscale(image1)
    gray2 = ensure_grayscale(image2)

    orb = cv2.ORB_create(nfeatures=n_features)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        print("  Error: Could not compute descriptors.")
        return None, None, status

    print(f"  Features: image1={len(kp1)}, image2={len(kp2)}")

    # Step 2: Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    status["total_matches"] = len(matches)

    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good_matches.append(m)

    status["good_matches"] = len(good_matches)
    print(f"  Matches: total={len(matches)}, good={len(good_matches)}")

    # Visualize matches
    match_vis = cv2.drawMatches(
        image1, kp1, image2, kp2, good_matches, None,
        matchColor=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
    )

    if len(good_matches) < min_matches:
        print(f"  Error: Not enough matches ({len(good_matches)} < {min_matches})")
        return None, match_vis, status

    # Step 3: Estimate homography with RANSAC
    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, reproj_thresh)
    inliers = int(mask.sum()) if mask is not None else 0
    status["inliers"] = inliers
    print(f"  Homography inliers: {inliers}/{len(good_matches)}")

    if H is None:
        print("  Error: Homography estimation failed.")
        return None, match_vis, status

    # Step 4: Warp and blend
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    panorama_width = w1 + w2
    panorama_height = max(h1, h2)

    # Warp image2 to image1's coordinate frame
    warped = cv2.warpPerspective(image2, H, (panorama_width, panorama_height))

    # Place image1 on the left side
    warped[0:h1, 0:w1] = image1

    # Crop black borders
    gray_panorama = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_panorama, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        warped = warped[y:y + h, x:x + w]

    status["success"] = True
    print(f"  Panorama size: {warped.shape[1]} x {warped.shape[0]}")
    return warped, match_vis, status


def stitch_multiple(images):
    """
    Stitch multiple images into a single panorama (left to right).

    Args:
        images: List of images (at least 2) in left-to-right order.

    Returns:
        tuple: (panorama, success)
    """
    if len(images) < 2:
        print("  Error: Need at least 2 images for stitching.")
        return None, False

    result = images[0]
    for i in range(1, len(images)):
        print(f"\n  Stitching image {i + 1}/{len(images)}...")
        result, _, status = stitch_images(result, images[i])
        if result is None:
            print(f"  Failed at image {i + 1}")
            return None, False

    return result, True
