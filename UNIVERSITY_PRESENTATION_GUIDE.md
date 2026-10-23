# 🎓 University Project Presentation & Defense Guide
## Automated Beach Litter Detection Using Deep Learning & Computer Vision

This guide is designed to help you confidently present and explain your project to your professor and university examination committee.

---

## 1. Problem Statement & Motivation
* **Environmental Context:** Coastal and marine plastic pollution has escalated into a global ecological crisis. Debris harms marine wildlife, damages coastal tourism, and degrades into toxic microplastics that enter the human food chain.
* **The Engineering Challenge:** Traditional beach monitoring relies on manual surveys, which are labor-intensive, slow, hazardous, and logistically expensive.
* **Our Solution:** An automated computer vision system utilizing single-stage deep learning object detection (YOLOv8) to locate, categorize, and count beach litter in real time from standard digital photography.

---

## 2. Project Objectives
1. Build an end-to-end local computer vision pipeline without relying on external cloud development services.
2. Train a convolutional neural network with hardware acceleration (NVIDIA GPU via PyTorch CUDA).
3. Detect multi-class coastal waste: **plastic bottles, cans, plastic bags/wrappers, plastic containers, styrofoam, cartons, and miscellaneous debris**.
4. Rigorously evaluate the model on an independent test set using industry-standard academic metrics: **Precision, Recall, mAP@50, and mAP@50–95**.
5. Build an intuitive testing interface that visualizes bounding boxes, confidence scores, and handles clean/empty beach scenarios gracefully.

---

## 3. Hardware & Local Architecture
* **Operating System:** Windows 11 Home (64-bit)
* **Processor (CPU):** Intel Core i5-14450HX (10 cores, 16 threads)
* **Graphics Processor (GPU):** NVIDIA GeForce RTX 3050 6GB Laptop GPU (CUDA 12.4 enabled)
* **Environment:** Isolated Python virtual environment (`.venv`) ensuring reproducibility and zero system dependency conflicts.

> **Professor Question:** *"Why train locally on an RTX 3050 rather than in Google Colab?"*  
> **Your Answer:** *"Local training guarantees complete data privacy, eliminates cloud session timeouts, allows custom low-level CUDA optimization, and enables direct edge deployment on local cleanup robots or drones without recurring cloud subscription costs."*

---

## 4. Dataset & Class Distribution
The model was trained and evaluated on **1,500 real-world litter images** containing **4,784 manual bounding box annotations**.

### Dataset Splits (Strictly Separated):
* **Training Set:** 1,050 images (70%) — used to adjust neural network weights.
* **Validation Set:** 300 images (20%) — used to tune hyperparameters and prevent overfitting during training.
* **Test Set:** 150 images (10%) — strictly withheld until final evaluation to ensure objective benchmarking.

### Object Classes:
1. `bottle` (Plastic water bottles, glass bottles)
2. `can` (Aluminum soda cans, food cans)
3. `plastic_bag` (Grocery bags, snack wrappers, packaging films)
4. `plastic_container` (Takeout containers, plastic tubs)
5. `styrofoam` (Foam cups, insulation chunks)
6. `carton` (Milk/juice cartons, cardboard boxes)
7. `other_litter` (Caps, straws, cups, broken fragments)

---

## 5. Model Architecture & Transfer Learning
* **Chosen Architecture:** Ultralytics YOLOv8 Nano (`yolov8n`)
  * **Parameters:** 3.1 million weights
  * **Computational Cost:** 8.6 GFLOPs (Gigafloating-point Operations)
  * **Inference Speed:** ~16.5 ms per image (equivalent to **~60 Frames Per Second** on the RTX 3050 GPU — suitable for real-time video streaming).
* **Transfer Learning:**
  * Rather than training from random scratch (which requires hundreds of thousands of images), we initialized the network with weights pretrained on the massive COCO (Common Objects in Context) dataset.
  * The network already understood foundational visual features (edges, curves, shadows, textures).
  * We then fine-tuned the classification head and feature pyramid specifically for coastal litter detection.

---

## 6. Training Procedure
* **Optimizer:** AdamW (adaptive learning rate `lr0=0.00091`, momentum=0.9, weight decay=0.0005)
* **Epochs:** 25 epochs
* **Batch Size:** 8 images per batch
* **Resolution:** $640 \times 640$ pixels
* **Loss Functions:**
  $$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{box}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{dfl}}$$
  * **Box Loss (CIoU):** Measures spatial bounding box alignment and aspect ratio consistency.
  * **Class Loss (BCE):** Binary cross-entropy assessing correct category identification.
  * **Distribution Focal Loss (DFL):** Refines precise sub-pixel boundary coordinates.

---

## 7. Actual Experimental Results (Benchmark on 150 Unseen Test Images)

| Metric | Overall Score | What This Means |
| :--- | :--- | :--- |
| **Precision (P)** | **26.42%** | Out of all predicted waste items, 26.4% were confirmed real litter. |
| **Recall (R)** | **29.77%** | Out of all actual litter objects on the beach, the model successfully detected 29.8%. |
| **mAP@50** | **24.15%** | Mean Average Precision at IoU threshold 0.50. |
| **mAP@50–95** | **18.16%** | Comprehensive benchmark averaged across 10 IoU thresholds (0.50 to 0.95). |

### Per-Class Test Performance:
* **`bottle`:** **`48.7%`** mAP@50 | **`59.2%`** Recall (Highest performing class due to distinctive cylindrical contours).
* **`can`:** **`36.8%`** mAP@50 | **`40.9%`** Recall (Strong detection on metallic reflective surfaces).
* **`plastic_bag`:** **`32.2%`** mAP@50 | **`41.5%`** Recall (Demonstrated **up to 91.9% confidence** on clear beach test samples).
* **`carton`:** **`13.4%`** mAP@50 | **`18.2%`** Recall.
* **`styrofoam`:** **`8.8%`** mAP@50 | **`28.6%`** Recall.
* **`other_litter`:** **`21.0%`** mAP@50 | **`20.1%`** Recall.

> [!NOTE]
> All evaluation plots (Precision-Recall curves, F1-confidence curves, and Confusion Matrices) are saved locally in:  
> `Beach-Litter-Detection/results/evaluation/benchmark/`

---

## 8. Live Demonstration Instructions (During Your Presentation)

### Step 1: Open PowerShell and enter the project folder
```powershell
cd c:\Users\chikh\OneDrive\Desktop\dev\raplica\Beach-Litter-Detection
```

### Step 2: Launch the Interactive Menu
```powershell
.\.venv\Scripts\python.exe main.py
```
* Press `1` to show your professor that the **NVIDIA RTX 3050 GPU and CUDA** are active.
* Press `2` to run the **Dataset Integrity Check** (shows 1,500 images, 0 corrupt, 0 data leakage).
* Press `4` to display the **Evaluation Benchmark Metrics** on the test set.
* Press `5` to run real-time detection on a sample photo.

### Step 3: Run Direct Command-Line Predictions
To run detection on a specific sample photo and show the annotated bounding boxes:
```powershell
.\.venv\Scripts\python.exe predict.py --source test_samples/000002.jpg --weights models/best.pt --conf 0.25
```
* **Output:** Detects `plastic_bag` with **91.9% confidence**.
* Open the resulting annotated image from `results/predictions/000002.jpg` to show the drawn bounding box.

To demonstrate how the system handles a clean beach:
```powershell
.\.venv\Scripts\python.exe predict.py --source test_samples/batch_10_000013.jpg --weights models/best.pt --conf 0.25
```
* **Output:** Explains clearly: *"NO LITTER DETECTED - The beach area in this photograph appears clean or objects are below the 25% threshold."*

---

## 9. Academic Limitations & Future Work (For Top Marks)

Professors appreciate students who understand the limitations of their model:

1. **Occlusion & Weathering:** Litter partially buried under sand or covered in sea foam has lower recall than pristine objects.
2. **Deformable Objects:** Plastic bags and squashed containers have non-rigid shapes that vary wildly in geometry compared to rigid objects like bottles and cans.
3. **Class Imbalance:** Generic litter fragments (`other_litter`) outnumbered specific categories like styrofoam, creating class imbalance.
4. **Future Enhancements:**
   * **Synthetic Data Augmentation:** Use diffusion models or CutMix to overlay litter onto diverse sand and wave textures.
   * **Temporal Video Tracking (ByteTRACK):** Pair YOLOv8 with a tracker to count unique litter items across video frames on an unmanned drone flight without double-counting.
   * **Edge Optimization (TensorRT):** Export `best.pt` to NVIDIA TensorRT FP16 for 2x faster inference on mobile edge devices.
