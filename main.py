"""
Beach Litter Detection - Master Interactive Controller
A beginner-friendly control center to inspect hardware, verify datasets,
train models, evaluate metrics, and run detections.
"""

import os
import sys
import subprocess
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass



def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_banner():
    print("""
======================================================================
     🏖️  BEACH LITTER DETECTION - COMPUTER VISION SYSTEM  🏖️
         Automated Coastal Debris Detection Using Deep Learning
======================================================================
""")


def check_system():
    print("\n" + "=" * 60)
    print("🖥️  SYSTEM & HARDWARE INSPECTION")
    print("=" * 60)
    print(f"Python Version: {sys.version}")

    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        has_cuda = torch.cuda.is_available()
        print(f"CUDA (GPU) Available: {has_cuda}")
        if has_cuda:
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Available VRAM: {vram:.2f} GB")
            print("🚀 Status: High-Performance GPU Acceleration Ready!")
        else:
            print("ℹ️  Status: Running on CPU (training will be slower).")
    except ImportError:
        print("[WARNING] PyTorch is not yet installed in this environment.")

    try:
        import ultralytics
        print(f"Ultralytics YOLO: Version {ultralytics.__version__}")
    except ImportError:
        print("[WARNING] Ultralytics YOLO is not yet installed.")

    input("\nPress [Enter] to return to the main menu...")


def verify_dataset():
    print("\n" + "=" * 60)
    print("📁 DATASET INTEGRITY & VALIDATION CHECK")
    print("=" * 60)

    yaml_file = Path("dataset.yaml")
    if not yaml_file.exists():
        print("[ERROR] dataset.yaml not found!")
        input("\nPress [Enter] to return...")
        return

    data_dir = Path("data")
    if not data_dir.exists():
        print("[ERROR] data/ folder not found. Please ensure the dataset is placed in 'data/'.")
        input("\nPress [Enter] to return...")
        return

    splits = ["train", "valid", "test"]
    split_counts = {}
    leakage_check = set()
    leakage_found = False

    for s in splits:
        img_dir = data_dir / s / "images"
        lbl_dir = data_dir / s / "labels"

        if not img_dir.exists():
            # Check alternative: data/train without /images subfolder
            if (data_dir / s).exists():
                img_dir = data_dir / s

        if img_dir.exists():
            images = [p for p in img_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
            labels = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []
            split_counts[s] = {"images": len(images), "labels": len(labels)}

            # Check for data leakage (duplicate filenames across splits)
            for img in images:
                if img.name in leakage_check:
                    print(f"  [WARNING] Potential data leakage! '{img.name}' appears in multiple splits.")
                    leakage_found = True
                leakage_check.add(img.name)
        else:
            split_counts[s] = {"images": 0, "labels": 0}

    print("\nSplit Summary:")
    print(f"  • Training set   : {split_counts.get('train', {}).get('images', 0)} images, {split_counts.get('train', {}).get('labels', 0)} label files")
    print(f"  • Validation set : {split_counts.get('valid', {}).get('images', 0)} images, {split_counts.get('valid', {}).get('labels', 0)} label files")
    print(f"  • Test set       : {split_counts.get('test', {}).get('images', 0)} images, {split_counts.get('test', {}).get('labels', 0)} label files")

    if not leakage_found:
        print("  ✅ Data Leakage Check: Passed! No duplicate image filenames across splits.")

    total_imgs = sum(v["images"] for v in split_counts.values())
    if total_imgs > 0:
        print(f"\n✅ Total Verified Images: {total_imgs}")
        print("Dataset is structured properly and ready for model training.")
    else:
        print("\n[NOTE] No images found yet in data/train/images, data/valid/images, data/test/images.")
        print("Follow Step 5 instructions to download and populate the dataset folder.")

    input("\nPress [Enter] to return to the main menu...")


def run_training():
    print("\n" + "=" * 60)
    print("🏋️ MODEL TRAINING")
    print("=" * 60)
    epochs = input("Enter number of epochs [default: 30]: ").strip() or "30"
    batch = input("Enter batch size [default: 8]: ").strip() or "8"

    cmd = [sys.executable, "train.py", "--epochs", epochs, "--batch", batch]
    print(f"\nRunning command: {' '.join(cmd)}\n")
    subprocess.run(cmd)
    input("\nPress [Enter] to return to the main menu...")


def run_evaluation():
    print("\n" + "=" * 60)
    print("📊 MODEL EVALUATION ON TEST SET")
    print("=" * 60)

    weights = input("Enter weights path [default: models/best.pt]: ").strip() or "models/best.pt"
    cmd = [sys.executable, "evaluate.py", "--weights", weights, "--split", "test"]
    print(f"\nRunning command: {' '.join(cmd)}\n")
    subprocess.run(cmd)
    input("\nPress [Enter] to return to the main menu...")


def run_prediction():
    print("\n" + "=" * 60)
    print("🔍 DETECT LITTER ON A SAMPLE IMAGE")
    print("=" * 60)

    source = input("Enter image path or folder [default: test_samples]: ").strip() or "test_samples"
    conf = input("Enter confidence threshold (0.1 - 0.9) [default: 0.25]: ").strip() or "0.25"

    cmd = [sys.executable, "predict.py", "--source", source, "--conf", conf]
    print(f"\nRunning command: {' '.join(cmd)}\n")
    subprocess.run(cmd)
    input("\nPress [Enter] to return to the main menu...")


def view_demo_summary():
    print("\n" + "=" * 70)
    print("🎓 UNIVERSITY PRESENTATION GUIDE - HOW TO EXPLAIN THIS PROJECT")
    print("=" * 70)
    print("""
1. PROBLEM STATEMENT:
   Marine plastic pollution threatens wildlife, enters human food chains,
   and disrupts coastal tourism. Traditional beach surveys require manual
   labor. Computer vision allows automated, rapid monitoring via drones or
   cameras.

2. METHODOLOGY:
   • Architecture: Ultralytics YOLOv8 (You Only Look Once).
   • Single-Stage Detector: Predicts bounding boxes and class probabilities
     in a single forward pass, making it fast enough for real-time video.
   • Transfer Learning: Started with weights pretrained on COCO (80 classes),
     then fine-tuned the model on specialized coastal litter images.

3. HARDWARE ACCELERATION:
   • Trained locally on an NVIDIA GeForce RTX 3050 Laptop GPU using PyTorch CUDA.
   • Batch size of 8 and 640x640 resolution balanced memory efficiency with
     detection precision.

4. EVALUATION CRITERIA:
   • Precision: Out of all predicted waste items, how many are real litter.
   • Recall: Out of all actual litter on the beach, how many were found.
   • mAP@50: Mean Average Precision at IoU 0.50 (standard benchmark).
   • mAP@50-95: Comprehensive benchmark averaging across strict IoU thresholds.

5. FUTURE IMPROVEMENTS:
   • Multi-spectral / drone aerial image integration.
   • Temporal tracking to avoid double-counting debris across video frames.
   • Edge deployment on embedded hardware (e.g., Raspberry Pi or Jetson Nano).
""")
    input("\nPress [Enter] to return to the main menu...")


def main():
    while True:
        clear_screen()
        print_banner()
        print("Select an option:")
        print("  [1] Check System & GPU Environment")
        print("  [2] Inspect & Validate Dataset Integrity")
        print("  [3] Train Deep Learning Model (train.py)")
        print("  [4] Evaluate Model on Unseen Test Set (evaluate.py)")
        print("  [5] Detect Litter on an Image (predict.py)")
        print("  [6] View University Presentation Guide")
        print("  [0] Exit")
        print("-" * 70)

        choice = input("Enter your choice (0-6): ").strip()

        if choice == "1":
            check_system()
        elif choice == "2":
            verify_dataset()
        elif choice == "3":
            run_training()
        elif choice == "4":
            run_evaluation()
        elif choice == "5":
            run_prediction()
        elif choice == "6":
            view_demo_summary()
        elif choice == "0":
            print("\nExiting Beach Litter Detection system. Goodbye!\n")
            break
        else:
            input("\nInvalid selection. Press [Enter] to try again...")


if __name__ == "__main__":
    main()
