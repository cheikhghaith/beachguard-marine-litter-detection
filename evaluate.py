"""
Beach Litter Detection - Model Evaluation Script
Evaluates the trained YOLO model on the separate test set and saves performance metrics.
"""

import sys
import json
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
    parser = argparse.ArgumentParser(description="Evaluate trained YOLO model on Beach Litter Test Set")
    parser.add_argument("--weights", type=str, default="models/best.pt", help="Path to trained model weights (.pt)")
    parser.add_argument("--data", type=str, default="dataset.yaml", help="Path to dataset.yaml configuration file")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate ('test' or 'val')")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution for evaluation")
    parser.add_argument("--project", type=str, default="results/evaluation", help="Directory to save evaluation results")
    parser.add_argument("--name", type=str, default="benchmark", help="Run folder name")
    return parser.parse_args()


def evaluate():
    args = parse_args()

    weights_path = Path(args.weights)
    yaml_path = Path(args.data)

    if not weights_path.is_file():
        print(f"[ERROR] Model weights '{args.weights}' not found!")
        print("Please train a model first using 'python train.py' or provide a valid weights file.")
        sys.exit(1)

    if not yaml_path.is_file():
        print(f"[ERROR] Dataset configuration '{args.data}' not found!")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics is not installed. Please run: pip install ultralytics")
        sys.exit(1)

    print("=" * 65)
    print("📈 EVALUATING TRAINED MODEL ON UNSEEN TEST DATA")
    print(f"   - Model Weights : {weights_path.resolve()}")
    print(f"   - Dataset Config: {yaml_path.resolve()}")
    print(f"   - Target Split  : {args.split.upper()} set")
    print("=" * 65)

    model = YOLO(str(weights_path))

    # Run validation on the specified split
    eval_target_dir = Path(args.project).resolve() / args.name
    eval_target_dir.mkdir(parents=True, exist_ok=True)

    results = model.val(
        data=str(yaml_path.resolve()),
        split=args.split,
        imgsz=args.imgsz,
        project=str(eval_target_dir.parent),
        name=eval_target_dir.name,
        exist_ok=True,
        plots=True,
        save_json=True,
        verbose=True
    )

    # Extract metrics
    box_metrics = results.box
    precision = float(box_metrics.mp)       # Mean Precision across all classes
    recall = float(box_metrics.mr)          # Mean Recall across all classes
    map50 = float(box_metrics.map50)        # mAP at IoU=0.50
    map50_95 = float(box_metrics.map)       # mAP at IoU=0.50:0.95

    print("\n" + "=" * 65)
    print("🏆 OBJECT DETECTION PERFORMANCE METRICS")
    print("=" * 65)
    print(f"  • Precision (P)    : {precision:.4f}  ({precision * 100:.2f}%)")
    print(f"  • Recall (R)       : {recall:.4f}  ({recall * 100:.2f}%)")
    print(f"  • mAP@50           : {map50:.4f}  ({map50 * 100:.2f}%)")
    print(f"  • mAP@50-95        : {map50_95:.4f}  ({map50_95 * 100:.2f}%)")
    print("=" * 65)

    print("\n📚 WHAT THESE METRICS MEAN (FOR YOUR PROFESSOR):")
    print("-" * 65)
    print("1. Precision : When the model predicts a piece of litter, how often is it right?")
    print("              A score of {:.1f}% means {:.1f}% of detected objects are actual litter.".format(
        precision * 100, precision * 100
    ))
    print("2. Recall    : Out of all the litter actually present on the beach, how much did it find?")
    print("              A score of {:.1f}% means the model successfully identified {:.1f}% of all items.".format(
        recall * 100, recall * 100
    ))
    print("3. mAP@50    : The standard benchmark. Measures average precision when predicted boxes")
    print("              overlap real boxes by at least 50% (IoU >= 0.50).")
    print("4. mAP@50-95 : The stricter comprehensive benchmark. Averages precision across IoU thresholds")
    print("              from 50% to 95% in steps of 5%.")
    print("-" * 65)

    # Save summary report
    eval_dir = eval_target_dir

    summary_data = {
        "weights": str(weights_path.resolve()),
        "dataset": str(yaml_path.resolve()),
        "split": args.split,
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "mAP_50": round(map50, 4),
            "mAP_50_95": round(map50_95, 4)
        },
        "per_class": {}
    }

    # Per-class metrics if available
    class_names = results.names
    for cls_id, cls_name in class_names.items():
        if cls_id < len(box_metrics.p):
            summary_data["per_class"][cls_name] = {
                "precision": round(float(box_metrics.p[cls_id]), 4),
                "recall": round(float(box_metrics.r[cls_id]), 4),
                "mAP_50": round(float(box_metrics.ap50[cls_id]), 4),
                "mAP_50_95": round(float(box_metrics.ap[cls_id]), 4)
            }

    json_report = eval_dir / "evaluation_metrics.json"
    with open(json_report, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    text_report = eval_dir / "evaluation_report.txt"
    with open(text_report, "w", encoding="utf-8") as f:
        f.write("BEACH LITTER DETECTION - TEST EVALUATION REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model Weights : {weights_path.resolve()}\n")
        f.write(f"Evaluated on  : {args.split.upper()} set\n\n")
        f.write(f"Precision     : {precision:.4f} ({precision * 100:.2f}%)\n")
        f.write(f"Recall        : {recall:.4f} ({recall * 100:.2f}%)\n")
        f.write(f"mAP@50        : {map50:.4f} ({map50 * 100:.2f}%)\n")
        f.write(f"mAP@50-95     : {map50_95:.4f} ({map50_95 * 100:.2f}%)\n\n")
        f.write("Per-Class Performance:\n")
        f.write("-" * 50 + "\n")
        for cls_name, vals in summary_data["per_class"].items():
            f.write(f"  {cls_name:<20} | P: {vals['precision']:.3f} | R: {vals['recall']:.3f} | mAP@50: {vals['mAP_50']:.3f}\n")

    print(f"\n✅ Evaluation report saved to: {text_report.resolve()}")
    print(f"✅ Metric JSON saved to: {json_report.resolve()}")
    print(f"🖼️ Evaluation plots (confusion matrix, PR curve, etc.) saved in: {eval_dir.resolve()}")


if __name__ == "__main__":
    evaluate()
