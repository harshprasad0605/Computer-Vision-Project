"""
Real-time Video Processing Module
===================================
Provides real-time webcam processing with selectable CV operations:
- Edge detection (Canny)
- Face detection (Haar cascade)
- Feature detection (ORB keypoints)
- Image filtering (Gaussian, bilateral)

Press 'q' to quit, or number keys (1-5) to switch modes.
"""

import cv2
import numpy as np

from config import (
    VIDEO_CAMERA_INDEX, VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT,
    CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD,
)
from modules.edge_detection import detect_edges_canny
from modules.object_detection import _get_cascade_path
from modules.filtering import apply_gaussian_blur, apply_bilateral_filter


MODES = {
    "1": "Original",
    "2": "Canny Edges",
    "3": "Face Detection",
    "4": "ORB Features",
    "5": "Gaussian Blur",
    "6": "Bilateral Filter",
}


def process_webcam(initial_mode="1", camera_index=None, save_output=None):
    """
    Open webcam and apply real-time computer vision processing.

    Controls:
        - Press 1-6 to switch processing modes
        - Press 'q' to quit
        - Press 's' to save current frame

    Args:
        initial_mode: Starting mode key (1-6).
        camera_index: Camera device index.
        save_output: If provided, saves output as a video file.

    Available modes:
        1 - Original (no processing)
        2 - Canny edge detection
        3 - Face detection
        4 - ORB feature detection
        5 - Gaussian blur
        6 - Bilateral filter
    """
    camera_index = camera_index if camera_index is not None else VIDEO_CAMERA_INDEX
    current_mode = initial_mode

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Cannot open webcam. Please check your camera connection.")
        print("  If you don't have a webcam, this mode requires a camera device.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_FRAME_HEIGHT)

    # Load face cascade for face detection mode
    try:
        face_cascade_path = _get_cascade_path("haarcascade_frontalface_default.xml")
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
    except FileNotFoundError:
        face_cascade = None
        print("  Warning: Face cascade not found. Face detection disabled.")

    # ORB detector for feature mode
    orb = cv2.ORB_create(nfeatures=500)

    # Video writer
    writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            save_output, fourcc, 20.0,
            (VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT)
        )

    print("\n" + "=" * 50)
    print("  Real-Time Video Processing")
    print("=" * 50)
    print("  Controls:")
    for key, name in MODES.items():
        print(f"    [{key}] {name}")
    print("    [q] Quit")
    print("    [s] Save current frame")
    print("=" * 50 + "\n")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame from webcam.")
            break

        # Process frame based on current mode
        if current_mode == "1":
            output = frame

        elif current_mode == "2":
            edges = detect_edges_canny(frame)
            output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        elif current_mode == "3":
            output = frame.copy()
            if face_cascade is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                for (x, y, w, h) in faces:
                    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(output, "Face", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        elif current_mode == "4":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            keypoints = orb.detect(gray, None)
            output = cv2.drawKeypoints(
                frame, keypoints, None, color=(0, 255, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )

        elif current_mode == "5":
            output = apply_gaussian_blur(frame)

        elif current_mode == "6":
            output = apply_bilateral_filter(frame)

        else:
            output = frame

        # Add mode indicator overlay
        mode_text = f"Mode: {MODES.get(current_mode, 'Unknown')}"
        cv2.putText(output, mode_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(output, "Press 1-6 to switch | q to quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("CV Toolkit - Real-Time Processing", output)

        if writer:
            writer.write(output)

        frame_count += 1

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif chr(key) in MODES:
            current_mode = chr(key)
            print(f"  Switched to: {MODES[current_mode]}")
        elif key == ord("s"):
            save_path = f"output/frame_{frame_count}.png"
            cv2.imwrite(save_path, output)
            print(f"  [SAVED] {save_path}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"\n  Processed {frame_count} frames.")
