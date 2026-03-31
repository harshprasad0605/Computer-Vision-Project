# Image Processing & Analysis Toolkit

A modular, command-line computer vision application built with Python and OpenCV. This toolkit demonstrates core computer vision concepts through practical implementations of image filtering, edge detection, segmentation, feature detection, object detection, image stitching, and real-time video processing.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
  - [1. Image Filtering](#1-image-filtering)
  - [2. Edge Detection](#2-edge-detection)
  - [3. Image Segmentation](#3-image-segmentation)
  - [4. Feature Detection](#4-feature-detection)
  - [5. Feature Matching](#5-feature-matching)
  - [6. Object Detection](#6-object-detection)
  - [7. Image Stitching](#7-image-stitching)
  - [8. Real-Time Video Processing](#8-real-time-video-processing)
  - [9. Full Demo](#9-full-demo)
- [Sample Outputs](#sample-outputs)
- [CV Concepts Covered](#cv-concepts-covered)
- [License](#license)

---

## Features

| Module | Techniques Implemented |
|--------|----------------------|
| **Image Filtering** | Gaussian blur, median filter, bilateral filter, custom convolution kernels (sharpen, emboss, edge enhance) |
| **Edge Detection** | Sobel operator (X, Y, magnitude), Canny edge detector, Laplacian of Gaussian |
| **Segmentation** | Otsu's thresholding, adaptive thresholding (mean, Gaussian), watershed, GrabCut |
| **Feature Detection** | Harris corner detection, ORB (Oriented FAST and Rotated BRIEF) |
| **Feature Matching** | Brute-force matching with Lowe's ratio test |
| **Object Detection** | Haar cascade (face/eye detection), contour-based shape detection |
| **Image Stitching** | Homography estimation, RANSAC, perspective warping, panorama creation |
| **Video Processing** | Real-time webcam processing with switchable CV modes |

---

## Project Structure

```
CV Project/
├── main.py                      # CLI entry point with argparse
├── config.py                    # Centralized configuration constants
├── generate_samples.py          # Script to create test images
├── requirements.txt             # Python dependencies
├── modules/
│   ├── __init__.py
│   ├── filtering.py             # Spatial domain filtering
│   ├── edge_detection.py        # Gradient & Laplacian edge detection
│   ├── segmentation.py          # Thresholding, watershed, GrabCut
│   ├── feature_detection.py     # Harris, ORB, feature matching
│   ├── object_detection.py      # Haar cascades, contour analysis
│   ├── stitching.py             # Panorama stitching via homography
│   └── video_processing.py      # Real-time webcam processing
├── utils/
│   ├── __init__.py
│   ├── io_utils.py              # Image loading/saving utilities
│   └── visualization.py         # Comparison plots & grid layouts
├── data/
│   └── sample_images/           # Test images (generated or user-provided)
├── output/                      # Generated results (auto-created)
└── report/
    └── project_report.md        # Detailed project report
```

---

## Prerequisites

- **Python**: 3.8 or higher
- **pip**: Python package manager
- **Webcam** (optional): Required only for real-time video processing mode

---

## Setup & Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Generate Sample Images

```bash
python generate_samples.py
```

This creates test images in `data/sample_images/`:
- `shapes.png` — Geometric shapes (circles, rectangles, triangles, stars)
- `gradients.png` — Gradients with sharp edges for edge detection testing
- `noisy.png` — Image with salt-and-pepper noise for filter testing
- `textured.png` — Checkerboard and stripe patterns for texture analysis

### Step 5: Verify Installation

```bash
python main.py --help
```

---

## Usage

All commands follow the pattern:

```bash
python main.py <command> --input <image_path> --method <method_name>
```

Results are saved to the `output/` directory automatically.

### 1. Image Filtering

```bash
# Compare all filters side-by-side
python main.py filter --input data/sample_images/noisy.png --method all

# Apply a specific filter
python main.py filter --input data/sample_images/shapes.png --method gaussian
python main.py filter --input data/sample_images/noisy.png --method median
python main.py filter --input data/sample_images/shapes.png --method bilateral
python main.py filter --input data/sample_images/shapes.png --method sharpen
python main.py filter --input data/sample_images/shapes.png --method emboss
```

### 2. Edge Detection

```bash
# Compare all edge detectors
python main.py edges --input data/sample_images/shapes.png --method all

# Specific edge detector
python main.py edges --input data/sample_images/shapes.png --method canny
python main.py edges --input data/sample_images/gradients.png --method sobel
python main.py edges --input data/sample_images/shapes.png --method laplacian
```

### 3. Image Segmentation

```bash
# Compare all segmentation methods
python main.py segment --input data/sample_images/shapes.png --method all

# Specific segmentation
python main.py segment --input data/sample_images/shapes.png --method otsu
python main.py segment --input data/sample_images/textured.png --method adaptive
python main.py segment --input data/sample_images/shapes.png --method watershed
python main.py segment --input data/sample_images/shapes.png --method grabcut
```

### 4. Feature Detection

```bash
# Compare Harris and ORB
python main.py features --input data/sample_images/textured.png --method all

# Specific detector
python main.py features --input data/sample_images/shapes.png --method harris
python main.py features --input data/sample_images/textured.png --method orb
```

### 5. Feature Matching

```bash
# Match features between two images
python main.py match --input1 data/sample_images/shapes.png --input2 data/sample_images/noisy.png
```

### 6. Object Detection

```bash
# Compare face and contour detection
python main.py detect --input data/sample_images/shapes.png --method all

# Face detection (works best with photos of people)
python main.py detect --input path/to/photo.jpg --method face

# Contour-based shape detection
python main.py detect --input data/sample_images/shapes.png --method contour
```

### 7. Image Stitching

```bash
# Stitch two overlapping images into a panorama
python main.py stitch --images image_left.jpg image_right.jpg

# Stitch multiple images
python main.py stitch --images img1.jpg img2.jpg img3.jpg
```

### 8. Real-Time Video Processing

```bash
# Start webcam with default mode
python main.py video

# Start with specific mode
python main.py video --mode 2   # Start with Canny edge detection
python main.py video --camera 1 # Use second camera
```

**Controls during video mode:**
| Key | Mode |
|-----|------|
| `1` | Original (no processing) |
| `2` | Canny Edge Detection |
| `3` | Face Detection |
| `4` | ORB Feature Detection |
| `5` | Gaussian Blur |
| `6` | Bilateral Filter |
| `q` | Quit |
| `s` | Save current frame |

### 9. Full Demo

```bash
# Run all operations on sample images
python main.py demo
```

This automatically runs filtering, edge detection, segmentation, feature detection, and object detection on the sample images, saving all results to `output/`.

---

## CV Concepts Covered

| Concept | Where Implemented |
|---------|------------------|
| Convolution & Spatial Filtering | `modules/filtering.py` — Gaussian, median, bilateral, custom kernels |
| Gradient Computation | `modules/edge_detection.py` — Sobel X/Y gradients |
| Non-Maximum Suppression | `modules/edge_detection.py` — Canny edge detector |
| Hysteresis Thresholding | `modules/edge_detection.py` — Canny double threshold |
| Laplacian (2nd Derivative) | `modules/edge_detection.py` — Laplacian edge detection |
| Otsu's Method | `modules/segmentation.py` — Automatic threshold selection |
| Adaptive Thresholding | `modules/segmentation.py` — Local neighbourhood thresholds |
| Morphological Operations | `modules/segmentation.py` — Watershed preprocessing |
| Distance Transform | `modules/segmentation.py` — Watershed marker generation |
| Watershed Algorithm | `modules/segmentation.py` — Marker-based segmentation |
| Graph Cuts (GrabCut) | `modules/segmentation.py` — Foreground extraction |
| Harris Corner Detection | `modules/feature_detection.py` — Structure tensor analysis |
| ORB Features | `modules/feature_detection.py` — FAST + BRIEF descriptors |
| Feature Matching | `modules/feature_detection.py` — Brute-force + ratio test |
| Haar-like Features | `modules/object_detection.py` — Viola-Jones face detection |
| Contour Analysis | `modules/object_detection.py` — Shape classification |
| Homography & RANSAC | `modules/stitching.py` — Panorama creation |
| Perspective Warping | `modules/stitching.py` — Image alignment |
| Real-time Processing | `modules/video_processing.py` — Webcam pipeline |

---

## License

This project is developed for educational purposes as part of the Computer Vision course.
