"""
Beach Litter Detection - Model Training Script
Trained using Ultralytics YOLOv8 with PyTorch and CUDA support.
"""

import os
import sys
import shutil
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
    parser = argparse.ArgumentParser(description="Train YOLO model on Beach Litter Dataset")
    parser.add_argument("--data", type=str, default="dataset.yaml", help="Path to dataset.yaml configuration file")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained base model (e.g. yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (8 is safe for 6GB RTX 3050 VRAM)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution for training")
    parser.add_argument("--device", type=str, default="", help="Device: '0' for GPU, 'cpu' for CPU (empty for auto-detect)")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker processes (4 is optimal on Windows)")
    parser.add_argument("--project", type=str, default="results/training", help="Directory where training runs are saved")
    parser.add_argument("--name", type=str, default="beach_litter_run", help="Run name")
    return parser.parse_args()


def check_environment():
    """Verify PyTorch and GPU availability."""
    try:
        import torch
    except ImportError:
        print("[ERROR] PyTorch is not installed. Please install dependencies from requirements.txt.")
        sys.exit(1)

    cuda_available = torch.cuda.is_available()
    print("=" * 60)
    print("🚀 HARDWARE ENVIRONMENT CHECK")
    print(f"  PyTorch Version : {torch.__version__}")
    print(f"  CUDA Available  : {cuda_available}")
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU Device      : {gpu_name}")
        print(f"  VRAM Memory     : {vram_gb:.2f} GB")
    else:
        print("  [NOTE] Training will run on CPU. (NVIDIA GPU not detected)")
    print("=" * 60)
    return "0" if cuda_available else "cpu"


def train():
    args = parse_args()
    auto_device = check_environment()
    device = args.device if args.device else auto_device

    # Ensure dataset.yaml exists
    yaml_path = Path(args.data)
    if not yaml_path.is_file():
        print(f"[ERROR] Dataset configuration '{args.data}' not found!")
        print("Please ensure dataset.yaml is present in the project directory.")
        sys.exit(1)

    # Import Ultralytics YOLO
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics is not installed. Please run: pip install ultralytics")
        sys.exit(1)

    print(f"\n📦 Loading base pretrained model: {args.model} ...")
    model = YOLO(args.model)

    print("\n🎯 Commencing Model Fine-Tuning:")
    print(f"   - Dataset Config : {yaml_path.resolve()}")
    print(f"   - Target Epochs  : {args.epochs}")
    print(f"   - Batch Size     : {args.batch}")
    print(f"   - Image Size     : {args.imgsz}")
    print(f"   - Training Device: {device.upper()}")
    print("-" * 60)

    try:
        results = model.train(
            data=str(yaml_path.resolve()),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=device,
            workers=args.workers,
            project=args.project,
            name=args.name,
            exist_ok=True,
            plots=True,
            save=True,
            verbose=True
        )
    except Exception as e:
        print(f"\n[ERROR] An error occurred during training: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    # Save best weights to models/best.pt
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    trained_best = Path(args.project) / args.name / "weights" / "best.pt"
    target_best = models_dir / "best.pt"

    if trained_best.exists():
        shutil.copy(trained_best, target_best)
        print(f"✅ Best model weights saved locally to: {target_best.resolve()}")
    else:
        print(f"[WARNING] Could not locate {trained_best}. Check {args.project} folder.")

    print(f"📊 Training curves, plots, and logs are saved in: {Path(args.project) / args.name}")
    print("Next step: Run 'python evaluate.py' to benchmark the model on unseen test images.")


if __name__ == "__main__":
    train()
