"""
Generate sample test images for the CV Toolkit demo.
Run this script to create test images in data/sample_images/.
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR


def create_geometric_shapes():
    """Create an image with various geometric shapes for testing."""
    img = np.ones((500, 700, 3), dtype=np.uint8) * 240  # Light gray bg

    # Gradient background
    for i in range(500):
        img[i, :, 0] = min(240, 200 + i // 10)
        img[i, :, 1] = min(240, 210 + i // 12)
        img[i, :, 2] = max(180, 240 - i // 8)

    # Circle
    cv2.circle(img, (120, 120), 80, (50, 120, 200), -1)
    cv2.circle(img, (120, 120), 80, (30, 80, 150), 3)

    # Rectangle
    cv2.rectangle(img, (280, 50), (430, 190), (80, 180, 80), -1)
    cv2.rectangle(img, (280, 50), (430, 190), (40, 120, 40), 3)

    # Triangle
    pts = np.array([[580, 190], [500, 50], [660, 50]], np.int32)
    cv2.fillPoly(img, [pts], (200, 80, 80))
    cv2.polylines(img, [pts], True, (150, 40, 40), 3)

    # Ellipse
    cv2.ellipse(img, (120, 350), (90, 50), 30, 0, 360, (200, 180, 50), -1)
    cv2.ellipse(img, (120, 350), (90, 50), 30, 0, 360, (150, 130, 20), 3)

    # Star-like polygon
    angles = np.linspace(0, 2 * np.pi, 11)[:-1]
    radii = [70, 35] * 5
    cx, cy = 360, 350
    star_pts = []
    for a, r in zip(angles, radii):
        x = int(cx + r * np.cos(a - np.pi / 2))
        y = int(cy + r * np.sin(a - np.pi / 2))
        star_pts.append([x, y])
    star_pts = np.array(star_pts, np.int32)
    cv2.fillPoly(img, [star_pts], (180, 100, 200))
    cv2.polylines(img, [star_pts], True, (120, 60, 140), 3)

    # Pentagon
    angles_p = np.linspace(0, 2 * np.pi, 6)[:-1]
    cx2, cy2 = 580, 350
    pent_pts = []
    for a in angles_p:
        x = int(cx2 + 60 * np.cos(a - np.pi / 2))
        y = int(cy2 + 60 * np.sin(a - np.pi / 2))
        pent_pts.append([x, y])
    pent_pts = np.array(pent_pts, np.int32)
    cv2.fillPoly(img, [pent_pts], (100, 200, 200))
    cv2.polylines(img, [pent_pts], True, (60, 150, 150), 3)

    # Add some text
    cv2.putText(img, "CV Toolkit Test Image", (180, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 2)

    return img


def create_gradient_image():
    """Create an image with gradients for edge detection testing."""
    img = np.zeros((400, 600, 3), dtype=np.uint8)

    # Horizontal gradient
    for x in range(600):
        img[:200, x] = [int(x / 600 * 255)] * 3

    # Vertical gradient
    for y in range(200, 400):
        img[y, :] = [int((y - 200) / 200 * 255)] * 3

    # Sharp edges
    cv2.rectangle(img, (150, 50), (450, 150), (255, 255, 255), -1)
    cv2.rectangle(img, (200, 250), (400, 350), (0, 0, 0), -1)

    # Circles with different intensities
    cv2.circle(img, (100, 300), 40, (200, 200, 200), -1)
    cv2.circle(img, (500, 300), 40, (100, 100, 100), -1)

    return img


def create_noisy_image():
    """Create an image with salt-and-pepper noise for filter testing."""
    img = create_geometric_shapes()

    # Add salt-and-pepper noise
    noise = np.random.random(img.shape[:2])
    img[noise < 0.02] = 0        # pepper
    img[noise > 0.98] = 255      # salt

    return img


def create_textured_image():
    """Create an image with textures and patterns."""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 200

    # Checkerboard pattern
    block = 40
    for i in range(0, 400, block):
        for j in range(0, 300, block):
            if (i // block + j // block) % 2 == 0:
                img[i:i + block, j:j + block] = [50, 50, 50]

    # Striped pattern on right side
    for j in range(300, 600, 10):
        if (j // 10) % 2 == 0:
            img[:, j:j + 10] = [150, 100, 50]

    # Add circles
    for r in range(20, 150, 30):
        cv2.circle(img, (450, 200), r, (0, 0, 200 - r), 2)

    return img


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Generating sample images...")

    images = {
        "shapes.png": create_geometric_shapes(),
        "gradients.png": create_gradient_image(),
        "noisy.png": create_noisy_image(),
        "textured.png": create_textured_image(),
    }

    for name, img in images.items():
        path = os.path.join(DATA_DIR, name)
        cv2.imwrite(path, img)
        print(f"  Created: {path}")

    print(f"\nDone! {len(images)} sample images created in {DATA_DIR}")


if __name__ == "__main__":
    main()
