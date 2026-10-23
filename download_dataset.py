"""
Beach Litter Detection - Dataset Download & Validation Assistant
Helps organize, extract, and validate beach litter dataset files locally.
"""

import os
import sys
import yaml
import shutil
import zipfile
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
    parser = argparse.ArgumentParser(description="Download & Validate Beach Litter Dataset")
    parser.add_argument("--zip", type=str, default="", help="Path to a downloaded dataset zip file to extract")
    parser.add_argument("--data-dir", type=str, default="data", help="Target dataset directory")
    parser.add_argument("--yaml", type=str, default="dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--api-key", type=str, default="", help="Optional Roboflow API key for direct automated download")
    return parser.parse_args()


def extract_zip(zip_path, target_dir):
    """Extract a downloaded dataset zip file and organize into data/."""
    zip_p = Path(zip_path)
    if not zip_p.is_file():
        print(f"[ERROR] Zip file '{zip_path}' not found!")
        return False

    print(f"\n📦 Extracting '{zip_p.name}' into '{target_dir}'...")
    target_p = Path(target_dir)
    target_p.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_p, 'r') as zip_ref:
        zip_ref.extractall(target_p)

    print("✅ Extraction complete!")

    # Check if files extracted into an extra nested folder or directly
    # Some zips contain train/valid/test directly, some contain data.yaml
    extracted_yaml = target_p / "data.yaml"
    if extracted_yaml.exists():
        # Copy or sync dataset.yaml if needed
        shutil.copy(extracted_yaml, Path("dataset.yaml"))
        print("✅ Synced dataset.yaml configuration from downloaded archive.")

    return True


def download_via_roboflow(api_key, target_dir):
    """Download dataset directly using roboflow Python SDK if an API key is provided."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[NOTE] Installing roboflow package for download...")
        os.system(f'"{sys.executable}" -m pip install roboflow')
        from roboflow import Roboflow

    print("\n🌐 Connecting to Roboflow Universe (Beach Waste Dataset)...")
    rf = Roboflow(api_key=api_key)
    try:
        project = rf.workspace("monash-ventz").project("beach-waste-vqths")
        version = project.version(1)
        dataset = version.download("yolov8", location=target_dir)
        print(f"✅ Successfully downloaded dataset to: {target_dir}")
        return True
    except Exception as e:
        print(f"[ERROR] Roboflow download failed: {e}")
        return False


def validate_dataset(data_dir="data", yaml_path="dataset.yaml"):
    """
    Comprehensive dataset validation:
    1. Images exist and can be opened without corruption
    2. Labels exist and match images
    3. Class IDs are within valid bounds
    4. Train, valid, and test sets are properly separated
    5. Data leakage check (duplicates across splits)
    """
    print("\n" + "=" * 65)
    print("🔍 RUNNING COMPREHENSIVE DATASET VALIDATION")
    print("=" * 65)

    data_path = Path(data_dir)
    yaml_p = Path(yaml_path)

    if not yaml_p.exists():
        print(f"[ERROR] '{yaml_path}' not found. Cannot determine class names.")
        return False

    with open(yaml_p, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    class_names = config.get("names", {})
    if isinstance(class_names, list):
        class_names = {i: name for i, name in enumerate(class_names)}
    num_classes = len(class_names)

    print(f"📋 Configured Classes ({num_classes} total):")
    for cid, cname in class_names.items():
        print(f"   [{cid}] {cname}")
    print("-" * 65)

    # Check splits
    splits = ["train", "valid", "test"]
    split_stats = {}
    seen_image_names = {}
    total_annotations = 0
    class_distribution = {cid: 0 for cid in class_names.keys()}
    issues_found = []

    try:
        from PIL import Image
    except ImportError:
        Image = None

    for split in splits:
        split_dir = data_path / split
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        # Allow flat directory structure as fallback (split/ directly containing images)
        if not img_dir.exists() and split_dir.exists():
            img_dir = split_dir

        if not img_dir.exists():
            split_stats[split] = {"images": 0, "labels": 0, "corrupt": 0, "unlabeled": 0}
            continue

        images = [p for p in img_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
        corrupt_count = 0
        unlabeled_count = 0
        valid_labels_count = 0

        for img_p in images:
            # Check for data leakage across splits
            if img_p.name in seen_image_names:
                issues_found.append(
                    f"DATA LEAKAGE: Image '{img_p.name}' is in both '{seen_image_names[img_p.name]}' and '{split}'!"
                )
            else:
                seen_image_names[img_p.name] = split

            # Test opening image
            if Image is not None:
                try:
                    with Image.open(img_p) as im:
                        im.verify()
                except Exception as e:
                    corrupt_count += 1
                    issues_found.append(f"CORRUPT IMAGE: '{img_p}' cannot be read ({e})")
                    continue

            # Check corresponding label file
            lbl_p = (lbl_dir / f"{img_p.stem}.txt") if lbl_dir.exists() else (img_dir / f"{img_p.stem}.txt")
            if not lbl_p.exists():
                unlabeled_count += 1
            else:
                valid_labels_count += 1
                try:
                    with open(lbl_p, "r") as lf:
                        for line_idx, line in enumerate(lf, 1):
                            parts = line.strip().split()
                            if not parts:
                                continue
                            cls_id = int(parts[0])
                            if cls_id not in class_names:
                                issues_found.append(
                                    f"INVALID CLASS: File '{lbl_p.name}' line {line_idx} has class {cls_id} not in dataset.yaml!"
                                )
                            else:
                                class_distribution[cls_id] += 1
                                total_annotations += 1

                            # Verify coordinate normalization (0.0 to 1.0)
                            coords = [float(x) for x in parts[1:]]
                            for c in coords:
                                if c < 0.0 or c > 1.0:
                                    issues_found.append(
                                        f"OUT OF BOUNDS: '{lbl_p.name}' has non-normalized coordinate {c}!"
                                    )
                except Exception as e:
                    issues_found.append(f"LABEL READ ERROR: '{lbl_p}' ({e})")

        split_stats[split] = {
            "images": len(images),
            "labels": valid_labels_count,
            "corrupt": corrupt_count,
            "unlabeled": unlabeled_count
        }

    # Print summary
    print("\n📊 DATASET SPLIT VERIFICATION TABLE:")
    print(f"{'Split':<12} | {'Images':<10} | {'Labels':<10} | {'Corrupt':<10} | {'Status'}")
    print("-" * 60)
    for split, stats in split_stats.items():
        status = "✅ OK" if stats["images"] > 0 and stats["corrupt"] == 0 else ("⚠️ Empty" if stats["images"] == 0 else "❌ Errors")
        print(f"{split.upper():<12} | {stats['images']:<10} | {stats['labels']:<10} | {stats['corrupt']:<10} | {status}")
    print("-" * 60)

    total_imgs = sum(s["images"] for s in split_stats.values())
    print(f"Total Images: {total_imgs} | Total Bounding Boxes: {total_annotations}")

    if total_annotations > 0:
        print("\n🏷️ CLASS DISTRIBUTION (BOUNDING BOX COUNTS):")
        for cid, count in class_distribution.items():
            pct = (count / total_annotations) * 100 if total_annotations > 0 else 0
            bar = "█" * int(pct // 4)
            print(f"   [{cid}] {class_names[cid]:<18} : {count:>5} annotations ({pct:>5.1f}%) {bar}")

    if issues_found:
        print("\n⚠️ ISSUES DETECTED:")
        for iss in issues_found[:10]:
            print(f"   - {iss}")
        if len(issues_found) > 10:
            print(f"   ... and {len(issues_found) - 10} more issues.")
        return False
    else:
        print("\n✅ ALL INTEGRITY CHECKS PASSED!")
        print("   - No corrupted images.")
        print("   - All bounding box coordinates normalized properly.")
        print("   - No data leakage detected between splits.")
        return True


def show_download_instructions():
    print("""
======================================================================
     📥 HOW TO GET THE BEACH WASTE DATASET LOCALLY
======================================================================

You can easily obtain the approved Beach Waste Dataset through either:

METHOD A: Direct Browser Download (No Coding Required)
1. In your web browser, open the dataset page:
   https://universe.roboflow.com/monash-ventz/beach-waste-vqths
2. Click the 'Download Dataset' button in the upper-right corner.
3. Select format: 'YOLOv8'
4. Choose 'Download zip to computer' and click Continue.
5. Once downloaded, pass the zip file to this script:
   python download_dataset.py --zip "C:\\path\\to\\downloaded_file.zip"
   (The script will automatically unpack and organize everything for you!)

METHOD B: Automated Download via API Key
If you have a free Roboflow account:
   python download_dataset.py --api-key YOUR_ROBOFLOW_API_KEY

METHOD C: Manual Folder Placement
Simply drop your images and labels directly into:
   data/train/images/  and  data/train/labels/
   data/valid/images/  and  data/valid/labels/
   data/test/images/   and  data/test/labels/
======================================================================
""")


def main():
    args = parse_args()

    if args.zip:
        extract_zip(args.zip, args.data_dir)
        validate_dataset(args.data_dir, args.yaml)
    elif args.api_key:
        if download_via_roboflow(args.api_key, args.data_dir):
            validate_dataset(args.data_dir, args.yaml)
    else:
        # Check if dataset already exists in data/
        data_p = Path(args.data_dir)
        has_data = any((data_p / s).exists() for s in ["train", "valid", "test"])
        if has_data:
            validate_dataset(args.data_dir, args.yaml)
        else:
            show_download_instructions()


if __name__ == "__main__":
    main()
