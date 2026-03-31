"""
Configuration constants for the CV Project.
Centralizes default parameters for all computer vision algorithms.
"""

import os

# ─────────────────────────────────────────────
# Directory Paths
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "sample_images")
HAARCASCADE_DIR = os.path.join(PROJECT_ROOT, "data", "haarcascades")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Filtering Defaults
# ─────────────────────────────────────────────
GAUSSIAN_KERNEL_SIZE = (5, 5)
GAUSSIAN_SIGMA = 1.0
MEDIAN_KERNEL_SIZE = 5
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# ─────────────────────────────────────────────
# Edge Detection Defaults
# ─────────────────────────────────────────────
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
SOBEL_KERNEL_SIZE = 3
LAPLACIAN_KERNEL_SIZE = 3

# ─────────────────────────────────────────────
# Segmentation Defaults
# ─────────────────────────────────────────────
ADAPTIVE_BLOCK_SIZE = 11
ADAPTIVE_C = 2
GRABCUT_ITER_COUNT = 5
GRABCUT_RECT_MARGIN = 50  # pixels from border for initial rectangle

# ─────────────────────────────────────────────
# Feature Detection Defaults
# ─────────────────────────────────────────────
HARRIS_BLOCK_SIZE = 2
HARRIS_KSIZE = 3
HARRIS_K = 0.04
HARRIS_THRESHOLD = 0.01  # fraction of max response
ORB_N_FEATURES = 500
FEATURE_MATCH_RATIO = 0.75  # Lowe's ratio test

# ─────────────────────────────────────────────
# Object Detection Defaults
# ─────────────────────────────────────────────
FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = (30, 30)
CONTOUR_THRESHOLD = 127
CONTOUR_MIN_AREA = 500

# ─────────────────────────────────────────────
# Image Stitching Defaults
# ─────────────────────────────────────────────
STITCH_REPROJECTION_THRESH = 5.0
STITCH_MIN_MATCH_COUNT = 10

# ─────────────────────────────────────────────
# Video Processing Defaults
# ─────────────────────────────────────────────
VIDEO_CAMERA_INDEX = 0
VIDEO_FRAME_WIDTH = 640
VIDEO_FRAME_HEIGHT = 480
