"""
Image Processing & Analysis Toolkit
====================================
A command-line computer vision application demonstrating core CV concepts.

Usage:
    python main.py <command> [options]

Commands:
    filter    - Apply image filtering (Gaussian, median, bilateral, custom)
    edges     - Detect edges (Sobel, Canny, Laplacian)
    segment   - Segment image (Otsu, adaptive, watershed, GrabCut)
    features  - Detect features (Harris corners, ORB)
    match     - Match features between two images
    detect    - Detect objects (faces, contours)
    stitch    - Stitch images into panorama
    video     - Real-time webcam processing
    demo      - Run full demonstration on sample images

Run 'python main.py <command> --help' for command-specific options.
"""

import argparse
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OUTPUT_DIR, DATA_DIR
from utils.io_utils import load_image, load_multiple_images, print_image_info, save_image
from utils.visualization import show_comparison, show_image_grid


def cmd_filter(args):
    """Apply image filtering operations."""
    from modules.filtering import (
        apply_gaussian_blur, apply_median_filter,
        apply_bilateral_filter, apply_custom_kernel,
        compare_all_filters,
    )

    image = load_image(args.input)
    print_image_info(image, os.path.basename(args.input))

    if args.method == "all":
        print("\n[*] Comparing all filters...")
        path = compare_all_filters(image, save_name=args.output)
        return

    print(f"\n[*] Applying {args.method} filter...")
    method_map = {
        "gaussian": lambda img: apply_gaussian_blur(img),
        "median": lambda img: apply_median_filter(img),
        "bilateral": lambda img: apply_bilateral_filter(img),
        "sharpen": lambda img: apply_custom_kernel(img, "sharpen"),
        "emboss": lambda img: apply_custom_kernel(img, "emboss"),
        "edge_enhance": lambda img: apply_custom_kernel(img, "edge_enhance"),
    }

    if args.method not in method_map:
        print(f"Unknown method: {args.method}")
        print(f"Available: {', '.join(method_map.keys())}, all")
        return

    result = method_map[args.method](image)
    show_comparison(
        image, result,
        title_original="Original",
        title_processed=f"{args.method.title()} Filter",
        save_name=args.output,
        subdirectory="filtering"
    )
    save_image(result, f"{args.method}_result.png", subdirectory="filtering")


def cmd_edges(args):
    """Detect edges in an image."""
    from modules.edge_detection import (
        detect_edges_sobel, detect_edges_canny,
        detect_edges_laplacian, compare_all_edges,
    )

    image = load_image(args.input)
    print_image_info(image, os.path.basename(args.input))

    if args.method == "all":
        print("\n[*] Comparing all edge detectors...")
        compare_all_edges(image, save_name=args.output)
        return

    print(f"\n[*] Applying {args.method} edge detection...")

    if args.method == "sobel":
        result = detect_edges_sobel(image)
        save_image(result["magnitude"], args.output, subdirectory="edges")
        show_comparison(
            image, result["magnitude"],
            title_processed="Sobel Edges",
            save_name=f"sobel_comparison.png",
            subdirectory="edges"
        )
    elif args.method == "canny":
        result = detect_edges_canny(image)
        save_image(result, args.output, subdirectory="edges")
        show_comparison(
            image, result,
            title_processed="Canny Edges",
            save_name=f"canny_comparison.png",
            subdirectory="edges"
        )
    elif args.method == "laplacian":
        result = detect_edges_laplacian(image)
        save_image(result, args.output, subdirectory="edges")
        show_comparison(
            image, result,
            title_processed="Laplacian Edges",
            save_name=f"laplacian_comparison.png",
            subdirectory="edges"
        )
    else:
        print(f"Unknown method: {args.method}. Use: sobel, canny, laplacian, all")


def cmd_segment(args):
    """Segment an image."""
    from modules.segmentation import (
        threshold_otsu, threshold_adaptive,
        watershed_segmentation, grabcut_segmentation,
        compare_all_segmentation,
    )

    image = load_image(args.input)
    print_image_info(image, os.path.basename(args.input))

    if args.method == "all":
        print("\n[*] Comparing all segmentation methods...")
        compare_all_segmentation(image, save_name=args.output)
        return

    print(f"\n[*] Applying {args.method} segmentation...")

    if args.method == "otsu":
        _, result = threshold_otsu(image)
    elif args.method == "adaptive":
        result = threshold_adaptive(image)
    elif args.method == "watershed":
        result, _ = watershed_segmentation(image)
    elif args.method == "grabcut":
        result, _ = grabcut_segmentation(image)
    else:
        print(f"Unknown method: {args.method}. Use: otsu, adaptive, watershed, grabcut, all")
        return

    save_image(result, args.output, subdirectory="segmentation")
    show_comparison(
        image, result,
        title_processed=f"{args.method.title()} Segmentation",
        save_name=f"{args.method}_comparison.png",
        subdirectory="segmentation"
    )


def cmd_features(args):
    """Detect features in an image."""
    from modules.feature_detection import (
        detect_harris_corners, detect_orb_features,
        compare_feature_detection,
    )

    image = load_image(args.input)
    print_image_info(image, os.path.basename(args.input))

    if args.method == "all":
        print("\n[*] Comparing feature detectors...")
        compare_feature_detection(image, save_name=args.output)
        return

    print(f"\n[*] Detecting {args.method} features...")

    if args.method == "harris":
        result, n = detect_harris_corners(image)
    elif args.method == "orb":
        result, kps, desc = detect_orb_features(image)
    else:
        print(f"Unknown method: {args.method}. Use: harris, orb, all")
        return

    save_image(result, args.output, subdirectory="features")
    show_comparison(
        image, result,
        title_processed=f"{args.method.upper()} Features",
        save_name=f"{args.method}_comparison.png",
        subdirectory="features"
    )


def cmd_match(args):
    """Match features between two images."""
    from modules.feature_detection import match_features

    print(f"\n[*] Matching features between images...")
    image1 = load_image(args.input1)
    image2 = load_image(args.input2)

    match_img, good_matches, kp1, kp2 = match_features(image1, image2)
    save_image(match_img, args.output, subdirectory="features")


def cmd_detect(args):
    """Detect objects in an image."""
    from modules.object_detection import (
        detect_faces, detect_faces_and_eyes,
        detect_contours, compare_object_detection,
    )

    image = load_image(args.input)
    print_image_info(image, os.path.basename(args.input))

    if args.method == "all":
        print("\n[*] Comparing detection methods...")
        compare_object_detection(image, save_name=args.output)
        return

    print(f"\n[*] Running {args.method} detection...")

    if args.method == "face":
        result, faces = detect_faces(image)
    elif args.method == "face_eyes":
        result, faces, eyes = detect_faces_and_eyes(image)
    elif args.method == "contour":
        result, info = detect_contours(image)
    else:
        print(f"Unknown method: {args.method}. Use: face, face_eyes, contour, all")
        return

    save_image(result, args.output, subdirectory="detection")
    show_comparison(
        image, result,
        title_processed=f"{args.method.title()} Detection",
        save_name=f"{args.method}_comparison.png",
        subdirectory="detection"
    )


def cmd_stitch(args):
    """Stitch images into a panorama."""
    from modules.stitching import stitch_images, stitch_multiple

    print(f"\n[*] Stitching {len(args.images)} images...")
    images = load_multiple_images(args.images)

    if len(images) == 2:
        panorama, match_vis, status = stitch_images(images[0], images[1])
        if match_vis is not None:
            save_image(match_vis, "stitch_matches.png", subdirectory="stitching")
    else:
        panorama, success = stitch_multiple(images)

    if panorama is not None:
        save_image(panorama, args.output, subdirectory="stitching")
        print("\n  Panorama created successfully!")
    else:
        print("\n  Stitching failed. Ensure images have sufficient overlap.")


def cmd_video(args):
    """Start real-time video processing."""
    from modules.video_processing import process_webcam

    print("\n[*] Starting real-time video processing...")
    process_webcam(
        initial_mode=args.mode,
        camera_index=args.camera,
    )


def cmd_demo(args):
    """Run a full demonstration using sample images."""
    print("=" * 60)
    print("  IMAGE PROCESSING & ANALYSIS TOOLKIT — FULL DEMO")
    print("=" * 60)

    # Find sample images
    sample_dir = DATA_DIR
    if not os.path.isdir(sample_dir):
        print(f"\nError: Sample images directory not found: {sample_dir}")
        print("Please add images to data/sample_images/")
        return

    samples = [f for f in os.listdir(sample_dir)
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not samples:
        print(f"\nError: No images found in {sample_dir}")
        return

    sample_path = os.path.join(sample_dir, samples[0])
    print(f"\n  Using sample image: {samples[0]}")
    image = load_image(sample_path)
    print_image_info(image, samples[0])

    # 1. Filtering
    print("\n" + "-" * 40)
    print("  1. IMAGE FILTERING")
    print("-" * 40)
    from modules.filtering import compare_all_filters
    compare_all_filters(image)

    # 2. Edge Detection
    print("\n" + "-" * 40)
    print("  2. EDGE DETECTION")
    print("-" * 40)
    from modules.edge_detection import compare_all_edges
    compare_all_edges(image)

    # 3. Segmentation
    print("\n" + "-" * 40)
    print("  3. IMAGE SEGMENTATION")
    print("-" * 40)
    from modules.segmentation import compare_all_segmentation
    compare_all_segmentation(image)

    # 4. Feature Detection
    print("\n" + "-" * 40)
    print("  4. FEATURE DETECTION")
    print("-" * 40)
    from modules.feature_detection import compare_feature_detection
    compare_feature_detection(image)

    # 5. Object Detection
    print("\n" + "-" * 40)
    print("  5. OBJECT DETECTION")
    print("-" * 40)
    from modules.object_detection import compare_object_detection
    compare_object_detection(image)

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE — Check the output/ directory for results")
    print("=" * 60)


def build_parser():
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="CV Toolkit",
        description="Image Processing & Analysis Toolkit — "
                    "A CLI for core computer vision operations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── filter ──
    p_filter = subparsers.add_parser("filter", help="Apply image filters")
    p_filter.add_argument("--input", "-i", required=True, help="Input image path")
    p_filter.add_argument("--method", "-m", default="all",
                          choices=["gaussian", "median", "bilateral",
                                   "sharpen", "emboss", "edge_enhance", "all"],
                          help="Filter method (default: all)")
    p_filter.add_argument("--output", "-o", default="filter_result.png",
                          help="Output filename")

    # ── edges ──
    p_edges = subparsers.add_parser("edges", help="Detect edges")
    p_edges.add_argument("--input", "-i", required=True, help="Input image path")
    p_edges.add_argument("--method", "-m", default="all",
                         choices=["sobel", "canny", "laplacian", "all"],
                         help="Edge detection method (default: all)")
    p_edges.add_argument("--output", "-o", default="edge_result.png",
                         help="Output filename")

    # ── segment ──
    p_seg = subparsers.add_parser("segment", help="Segment image")
    p_seg.add_argument("--input", "-i", required=True, help="Input image path")
    p_seg.add_argument("--method", "-m", default="all",
                       choices=["otsu", "adaptive", "watershed", "grabcut", "all"],
                       help="Segmentation method (default: all)")
    p_seg.add_argument("--output", "-o", default="segment_result.png",
                       help="Output filename")

    # ── features ──
    p_feat = subparsers.add_parser("features", help="Detect features")
    p_feat.add_argument("--input", "-i", required=True, help="Input image path")
    p_feat.add_argument("--method", "-m", default="all",
                        choices=["harris", "orb", "all"],
                        help="Feature detection method (default: all)")
    p_feat.add_argument("--output", "-o", default="feature_result.png",
                        help="Output filename")

    # ── match ──
    p_match = subparsers.add_parser("match", help="Match features between images")
    p_match.add_argument("--input1", "-i1", required=True, help="First image")
    p_match.add_argument("--input2", "-i2", required=True, help="Second image")
    p_match.add_argument("--output", "-o", default="match_result.png",
                         help="Output filename")

    # ── detect ──
    p_det = subparsers.add_parser("detect", help="Detect objects")
    p_det.add_argument("--input", "-i", required=True, help="Input image path")
    p_det.add_argument("--method", "-m", default="all",
                       choices=["face", "face_eyes", "contour", "all"],
                       help="Detection method (default: all)")
    p_det.add_argument("--output", "-o", default="detect_result.png",
                       help="Output filename")

    # ── stitch ──
    p_stitch = subparsers.add_parser("stitch", help="Stitch images into panorama")
    p_stitch.add_argument("--images", "-i", nargs="+", required=True,
                          help="Image paths (at least 2)")
    p_stitch.add_argument("--output", "-o", default="panorama.png",
                          help="Output filename")

    # ── video ──
    p_video = subparsers.add_parser("video", help="Real-time video processing")
    p_video.add_argument("--mode", "-m", default="1",
                         choices=["1", "2", "3", "4", "5", "6"],
                         help="Initial processing mode (1-6)")
    p_video.add_argument("--camera", "-c", type=int, default=0,
                         help="Camera device index")

    # ── demo ──
    subparsers.add_parser("demo", help="Run full demonstration")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start = time.time()

    command_map = {
        "filter": cmd_filter,
        "edges": cmd_edges,
        "segment": cmd_segment,
        "features": cmd_features,
        "match": cmd_match,
        "detect": cmd_detect,
        "stitch": cmd_stitch,
        "video": cmd_video,
        "demo": cmd_demo,
    }

    try:
        command_map[args.command](args)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - start
    print(f"\n  Completed in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
