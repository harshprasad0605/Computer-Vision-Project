"""
Object Detection Module
========================
Implements object detection using:
- Haar cascade classifiers (face and eye detection)
- Contour-based shape detection

Haar cascades use a trained classifier with Haar-like features
and a cascade of boosted classifiers for efficient detection.
Contour-based detection uses thresholding and contour analysis
to identify shapes in an image.
"""

import os
import cv2
import numpy as np

from config import (
    HAARCASCADE_DIR,
    FACE_SCALE_FACTOR, FACE_MIN_NEIGHBORS, FACE_MIN_SIZE,
    CONTOUR_THRESHOLD, CONTOUR_MIN_AREA,
)
from utils.io_utils import ensure_grayscale, ensure_color
from utils.visualization import show_image_grid, show_comparison


def _get_cascade_path(cascade_name):
    """
    Get the path to a Haar cascade XML file.
    Falls back to OpenCV's built-in cascades if local copy not found.
    """
    # Try local data directory first
    local_path = os.path.join(HAARCASCADE_DIR, cascade_name)
    if os.path.isfile(local_path):
        return local_path

    # Fall back to OpenCV's built-in cascades
    cv2_data = os.path.join(os.path.dirname(cv2.__file__), "data")
    builtin_path = os.path.join(cv2_data, cascade_name)
    if os.path.isfile(builtin_path):
        return builtin_path

    raise FileNotFoundError(
        f"Cascade file not found: {cascade_name}\n"
        f"  Searched: {local_path}\n"
        f"  Searched: {builtin_path}"
    )


def detect_faces(image, scale_factor=None, min_neighbors=None, min_size=None):
    """
    Detect faces using Haar cascade classifier.

    The Viola-Jones algorithm uses Haar-like features (rectangular
    patterns encoding intensity differences) evaluated using integral
    images for speed. A cascade of AdaBoost classifiers rejects
    non-face regions early, achieving real-time performance.

    Args:
        image: Input image (BGR).
        scale_factor: Scale reduction factor between pyramid levels.
        min_neighbors: Minimum number of detections to validate a face.
        min_size: Minimum face size as (width, height).

    Returns:
        tuple: (result_image, faces)
            - result_image: Image with rectangles drawn around faces
            - faces: Array of (x, y, w, h) for each detected face
    """
    scale_factor = scale_factor or FACE_SCALE_FACTOR
    min_neighbors = min_neighbors or FACE_MIN_NEIGHBORS
    min_size = min_size or FACE_MIN_SIZE

    cascade_path = _get_cascade_path("haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)

    gray = ensure_grayscale(image)
    result = image.copy()

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Label
        cv2.putText(result, "Face", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    print(f"  Faces detected: {len(faces)}")
    return result, faces


def detect_faces_and_eyes(image):
    """
    Detect faces and eyes within each face region.

    Args:
        image: Input BGR image.

    Returns:
        tuple: (result_image, faces, eyes_per_face)
    """
    cascade_face = _get_cascade_path("haarcascade_frontalface_default.xml")
    cascade_eye = _get_cascade_path("haarcascade_eye.xml")

    face_cascade = cv2.CascadeClassifier(cascade_face)
    eye_cascade = cv2.CascadeClassifier(cascade_eye)

    gray = ensure_grayscale(image)
    result = image.copy()

    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    eyes_per_face = []

    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Detect eyes within the face region
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = result[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
        eyes_per_face.append(len(eyes))

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh),
                          (255, 0, 0), 2)

    print(f"  Faces: {len(faces)}, Eyes: {sum(eyes_per_face)}")
    return result, faces, eyes_per_face


def detect_contours(image, threshold=None, min_area=None):
    """
    Detect and analyze contours in an image.

    Contour detection finds the boundaries of objects by tracing
    connected components in a binary image. Each contour is analyzed
    for properties like area, perimeter, and shape approximation.

    Args:
        image: Input image.
        threshold: Binary threshold value.
        min_area: Minimum contour area to consider.

    Returns:
        tuple: (result_image, contour_info)
            - result_image: Image with contours drawn
            - contour_info: List of dicts with contour properties
    """
    threshold = threshold or CONTOUR_THRESHOLD
    min_area = min_area or CONTOUR_MIN_AREA

    gray = ensure_grayscale(image)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    result = ensure_color(image.copy()) if len(image.shape) == 2 else image.copy()
    contour_info = []

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        n_vertices = len(approx)

        # Classify shape
        if n_vertices == 3:
            shape = "Triangle"
        elif n_vertices == 4:
            # Check aspect ratio for square vs rectangle
            x, y, w, h = cv2.boundingRect(approx)
            aspect = float(w) / h
            shape = "Square" if 0.85 <= aspect <= 1.15 else "Rectangle"
        elif n_vertices == 5:
            shape = "Pentagon"
        elif n_vertices > 5:
            shape = "Circle" if n_vertices > 8 else "Polygon"
        else:
            shape = "Unknown"

        # Draw contour and label
        color = (
            int(np.random.randint(50, 255)),
            int(np.random.randint(50, 255)),
            int(np.random.randint(50, 255)),
        )
        cv2.drawContours(result, [approx], -1, color, 2)
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(result, shape, (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        contour_info.append({
            "index": i,
            "shape": shape,
            "vertices": n_vertices,
            "area": area,
            "perimeter": round(perimeter, 2),
        })

    print(f"  Contours detected: {len(contour_info)}")
    for info in contour_info[:10]:  # Print first 10
        print(f"    {info['shape']}: area={info['area']:.0f}, "
              f"perimeter={info['perimeter']}")

    return result, contour_info


def compare_object_detection(image, save_name="detection_comparison.png"):
    """
    Compare face detection and contour detection results.

    Args:
        image: Input BGR image.
        save_name: Output filename.

    Returns:
        str: Path to saved comparison.
    """
    face_img, _ = detect_faces(image)
    contour_img, _ = detect_contours(image)

    images = [image, face_img, contour_img]
    titles = ["Original", "Face Detection", "Contour Detection"]

    return show_image_grid(
        images, titles, grid_cols=3,
        save_name=save_name, subdirectory="detection",
        main_title="Object Detection Comparison"
    )
