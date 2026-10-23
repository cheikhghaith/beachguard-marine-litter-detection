"""
Convert existing local TACO litter dataset (1,500 images + annotations.json)
into YOLO format split into train, valid, and test sets.
"""

import os
import sys
import json
import shutil
import random
from pathlib import Path
from PIL import Image

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass



def convert_local_dataset(
    source_dir=r"C:\Users\chikh\OneDrive\Desktop\computer vision\beach-litter-detection\datasets\raw\taco\data",
    output_dir=r"c:\Users\chikh\OneDrive\Desktop\dev\raplica\Beach-Litter-Detection\data",
    yaml_path=r"c:\Users\chikh\OneDrive\Desktop\dev\raplica\Beach-Litter-Detection\dataset.yaml"
):
    source_p = Path(source_dir)
    ann_file = source_p / "annotations.json"

    if not ann_file.exists():
        print(f"[ERROR] annotations.json not found in {source_dir}")
        return False

    print("=" * 65)
    print("🔄 CONVERTING LOCAL LITTER DATASET TO YOLO FORMAT")
    print(f"  • Source Directory : {source_p}")
    print(f"  • Output Directory : {output_dir}")
    print("=" * 65)

    with open(ann_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Simplified superclasses for litter detection
    # Map COCO category IDs to 6 clean classes
    target_classes = {
        0: "bottle",            # plastic bottles, glass bottles
        1: "can",               # drink cans, food cans
        2: "plastic_bag",       # plastic bags, wrappers
        3: "plastic_container", # food containers, boxes
        4: "styrofoam",         # styrofoam pieces
        5: "carton",            # drink cartons, boxes
        6: "other_litter"       # cups, lids, straws, etc.
    }

    # Map COCO categories to target class IDs
    cat_to_target = {}
    for cat in coco["categories"]:
        cid = cat["id"]
        cname = cat["name"].lower()
        supercat = cat["supercategory"].lower()

        if "bottle" in cname or "bottle" in supercat:
            cat_to_target[cid] = 0
        elif "can" in cname or "can" in supercat:
            cat_to_target[cid] = 1
        elif "bag" in cname or "wrapper" in cname or "plastic bag" in supercat:
            cat_to_target[cid] = 2
        elif "container" in cname or "container" in supercat:
            cat_to_target[cid] = 3
        elif "styrofoam" in cname or "styrofoam" in supercat:
            cat_to_target[cid] = 4
        elif "carton" in cname or "carton" in supercat:
            cat_to_target[cid] = 5
        else:
            cat_to_target[cid] = 6

    # Build image map
    img_map = {}
    for img in coco["images"]:
        img_map[img["id"]] = img

    # Group annotations by image_id
    ann_by_img = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in ann_by_img:
            ann_by_img[img_id] = []
        ann_by_img[img_id].append(ann)

    # Filter images that actually exist on disk and have annotations
    valid_images = []
    for img_id, img_info in img_map.items():
        rel_path = img_info["file_name"]
        full_path = source_p / rel_path
        if full_path.is_file() and img_id in ann_by_img:
            valid_images.append(img_info)

    print(f"Found {len(valid_images)} existing annotated images.")

    # Shuffle and split: 70% Train, 20% Valid, 10% Test
    random.seed(42)
    random.shuffle(valid_images)

    n_total = len(valid_images)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.20)

    splits = {
        "train": valid_images[:n_train],
        "valid": valid_images[n_train:n_train + n_val],
        "test": valid_images[n_train + n_val:]
    }

    print(f"Splits: Train={len(splits['train'])}, Valid={len(splits['valid'])}, Test={len(splits['test'])}")

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    class_counts = {cid: 0 for cid in target_classes}

    for split_name, img_list in splits.items():
        img_out = out_p / split_name / "images"
        lbl_out = out_p / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_info in img_list:
            src_img_file = source_p / img_info["file_name"]
            # Generate unique filename incorporating the batch folder (e.g. batch_1_000000.jpg)
            unique_name = img_info["file_name"].replace("/", "_").replace("\\", "_")
            stem = Path(unique_name).stem
            dest_img_file = img_out / unique_name

            # Copy image
            shutil.copy2(src_img_file, dest_img_file)

            # Get image dimensions
            img_w = img_info["width"]
            img_h = img_info["height"]

            # Write YOLO label file
            lbl_file = lbl_out / f"{stem}.txt"
            with open(lbl_file, "w", encoding="utf-8") as lf:
                for ann in ann_by_img.get(img_info["id"], []):
                    cat_id = ann["category_id"]
                    if cat_id not in cat_to_target:
                        continue

                    target_cls = cat_to_target[cat_id]
                    class_counts[target_cls] += 1

                    # COCO bbox: [x_min, y_min, width, height]
                    bbox = ann["bbox"]
                    x_min, y_min, bw, bh = bbox

                    if bw <= 0 or bh <= 0 or img_w <= 0 or img_h <= 0:
                        continue

                    # YOLO format: class_id x_center y_center width height (normalized 0-1)
                    x_center = (x_min + bw / 2.0) / img_w
                    y_center = (y_min + bh / 2.0) / img_h
                    norm_w = bw / img_w
                    norm_h = bh / img_h

                    # Clip to 0.0 - 1.0
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    norm_w = max(0.0, min(1.0, norm_w))
                    norm_h = max(0.0, min(1.0, norm_h))

                    lf.write(f"{target_cls} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

    # Update dataset.yaml
    yaml_content = f"""# Ultralytics YOLO Dataset Configuration for Beach & Marine Litter Detection
path: {out_p.resolve()}
train: train/images
val: valid/images
test: test/images

nc: {len(target_classes)}
names:
"""
    for cid, cname in target_classes.items():
        yaml_content += f"  {cid}: {cname}\n"

    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write(yaml_content)

    print("\n✅ DATASET PREPARATION COMPLETE!")
    print(f"Updated config: {yaml_path}")
    print("\nClass Annotation Breakdown:")
    for cid, cname in target_classes.items():
        print(f"  [{cid}] {cname:<18}: {class_counts[cid]} annotations")

    # Also copy 3 sample images into test_samples for instant testing
    test_samples_dir = Path(output_dir).parent / "test_samples"
    test_samples_dir.mkdir(parents=True, exist_ok=True)
    sample_imgs = list((out_p / "test" / "images").glob("*.jpg"))[:3]
    for s in sample_imgs:
        shutil.copy2(s, test_samples_dir / s.name)
    print(f"\nCopied {len(sample_imgs)} test images into 'test_samples/' for quick testing.")

    return True


if __name__ == "__main__":
    convert_local_dataset()
