"""
Beach Litter Detection - Inference & Testing Script
Runs detection on beach images, draws bounding boxes, displays confidence scores,
and saves the annotated output images.
"""

import sys
import argparse
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass



def parse_args():
    parser = argparse.ArgumentParser(description="Detect litter in beach images using trained YOLO model")
    parser.add_argument("--source", type=str, default="test_samples", help="Path to an image file or folder of images")
    parser.add_argument("--weights", type=str, default="models/best.pt", help="Path to model weights (.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0.0 to 1.0)")
    parser.add_argument("--save-dir", type=str, default="results/predictions", help="Directory to save annotated images")
    return parser.parse_args()


def predict():
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.is_file():
        # Fallback to yolov8n.pt if best.pt hasn't been trained yet
        fallback = Path("models/yolov8n.pt")
        if fallback.is_file():
            print(f"[NOTE] '{args.weights}' not found. Using pretrained base model: {fallback}")
            weights_path = fallback
        else:
            print(f"[ERROR] Model weights '{args.weights}' not found!")
            print("Please train a model first using 'python train.py' or pass valid weights via --weights.")
            sys.exit(1)

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"[ERROR] Source path '{args.source}' does not exist!")
        print("Please place a test image in 'test_samples/' or specify a valid file path.")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics is not installed. Please run: pip install ultralytics")
        sys.exit(1)

    print("=" * 65)
    print("🔍 BEACH LITTER DETECTION INFERENCE")
    print(f"  • Model Weights       : {weights_path.resolve()}")
    print(f"  • Input Source        : {source_path.resolve()}")
    print(f"  • Confidence Threshold: {args.conf}")
    print(f"  • Output Directory    : {Path(args.save_dir).resolve()}")
    print("=" * 65)

    model = YOLO(str(weights_path))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Run inference
    results = model.predict(
        source=str(source_path),
        conf=args.conf,
        save=True,
        project=str(save_dir.parent),
        name=save_dir.name,
        exist_ok=True,
        verbose=False
    )

    total_images = len(results)
    total_detections = 0

    print(f"\nProcessing {total_images} image(s)...\n")

    for idx, r in enumerate(results, 1):
        img_name = Path(r.path).name
        boxes = r.boxes
        num_detected = len(boxes)
        total_detections += num_detected

        print(f"[{idx}/{total_images}] Image: {img_name}")
        print("-" * 50)

        if num_detected == 0:
            print("  ℹ️  NO LITTER DETECTED")
            print("      Explanation: No waste objects were detected above the")
            print(f"      confidence threshold of {args.conf * 100:.0f}%.")
            print("      This could mean:")
            print("        1. The beach area in this photograph is clean.")
            print("        2. Any litter present is too distant, blurred, or occluded.")
            print("        3. Try lowering the threshold: python predict.py --conf 0.15")
        else:
            print(f"  🎯 Found {num_detected} litter item(s):")
            print(f"  {'Class':<20} | {'Confidence':<12} | {'Bounding Box [x1, y1, x2, y2]'}")
            print("  " + "-" * 55)

            for b in boxes:
                cls_id = int(b.cls[0].item())
                cls_name = r.names.get(cls_id, f"Class {cls_id}")
                conf = float(b.conf[0].item())
                xyxy = [round(float(coord), 1) for coord in b.xyxy[0].tolist()]

                print(f"  {cls_name:<20} | {conf * 100:>6.1f}%     | {xyxy}")

        print()

    print("=" * 65)
    print(f"✨ INFERENCE SUMMARY: Detected {total_detections} object(s) across {total_images} image(s).")
    print(f"📁 Annotated images saved in: {save_dir.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    predict()
