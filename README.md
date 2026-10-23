# 🏖️ Beach Litter Detection Using Computer Vision & Deep Learning

An end-to-end local computer vision system for detecting and localizing coastal litter and marine debris (such as plastic bottles, containers, bags, cans, and foam) using YOLO (You Only Look Once) deep learning models.

---

## 📌 Project Overview & University Objectives

Coastal plastic pollution is a major environmental threat to marine ecosystems. Manual beach cleanups and surveys are labor-intensive, slow, and expensive. This project demonstrates how modern deep learning object detection algorithms can automatically locate and classify waste items along shores and beaches from standard photography.

### Key Features:
* **Object Detection & Localization:** Predicts precise bounding boxes around litter items with classification labels and confidence scores.
* **Local Hardware Acceleration:** Configured for local GPU acceleration (NVIDIA RTX 3050 with CUDA) without relying on cloud services or third-party web platforms.
* **Rigorous Evaluation:** Measures real performance via Precision, Recall, mAP@50, mAP@50–95, and Confusion Matrices on unseen test data.
* **Beginner-Friendly Interactive Interface:** Includes an interactive menu (`main.py`) and modular standalone scripts (`train.py`, `evaluate.py`, `predict.py`).

---

## 📁 Directory Structure & File Guide

```text
Beach-Litter-Detection/
├── .venv/                 # Isolated Python virtual environment
├── data/                  # Dataset directory
│   ├── train/             # Training images and YOLO bounding box labels
│   ├── valid/             # Validation images and labels (used during training)
│   └── test/              # Unseen test images and labels (used for final evaluation)
├── models/                # Saved model weights (pretrained & custom trained best.pt)
├── results/               # Training curves, confusion matrices, and test predictions
│   ├── training/          # Epoch loss curves, PR curves, and training logs
│   ├── evaluation/        # Test set evaluation metrics and validation plots
│   └── predictions/       # Output images with drawn bounding boxes and labels
├── dataset.yaml           # YOLO configuration defining paths and litter classes
├── requirements.txt       # List of required Python packages
├── train.py               # Script to fine-tune the YOLO model on beach litter data
├── evaluate.py            # Script to benchmark the trained model on test images
├── predict.py             # Script to run detection on single or multiple images
├── main.py                # Interactive master controller for the entire pipeline
└── README.md              # Project documentation and student guide
```

### Purpose of Key Files:
* **`dataset.yaml`**: The bridge between YOLO and our dataset. Specifies where images are located and assigns an integer ID to each class name.
* **`train.py`**: Loads a lightweight pretrained model (`yolov8n.pt`), applies transfer learning, and trains the neural network to identify beach debris while saving the best checkpoint to `models/best.pt`.
* **`evaluate.py`**: Runs inference on the test split that the model has never seen before, calculating objective metrics (mAP, Precision, Recall).
* **`predict.py`**: The application demonstration tool. Takes any input photo, runs inference, draws bounding boxes, and prints detections.
* **`main.py`**: Master script providing an easy numeric menu (1 to 6) for running any phase of the project with a single command.

---

## ⚙️ Environment Setup & Installation

All commands are run locally on your Windows machine using PowerShell.

### 1. Activate the Virtual Environment
Open PowerShell inside this directory (`Beach-Litter-Detection`) and run:
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell asks for execution policy permission, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### 2. Verify GPU Acceleration
To verify that PyTorch sees your NVIDIA GeForce RTX 3050 GPU:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 🚀 How to Run the Project

### Option A: The Interactive Menu (Recommended for Beginners)
Run the master script:
```powershell
python main.py
```
This opens an interactive menu where you can press a number to check the environment, inspect the dataset, train, evaluate, or test a photo.

### Option B: Running Individual Scripts

#### 1. Train the Model
```powershell
python train.py --epochs 30 --batch 8 --imgsz 640
```
* The training script automatically selects your NVIDIA GPU.
* Training logs and curves are saved to `results/training/`.
* The best-performing model weights are saved to `models/best.pt`.

#### 2. Evaluate on Unseen Test Data
```powershell
python evaluate.py --weights models/best.pt
```
* Generates the confusion matrix, precision-recall curve, and prints mAP scores.

#### 3. Test on a New Beach Photo
```powershell
python predict.py --source test_samples/sample1.jpg --weights models/best.pt --conf 0.25
```
* Draws bounding boxes around detected waste items.
* Saves the output image with detections to `results/predictions/`.
* Explains clearly if no objects are detected.

---

## 📊 Understanding the Key Metrics (For Your University Demo)

When presenting this project to your professor, explain these core computer vision concepts:

1. **Bounding Box (Localization):**
   Defined by coordinates $[x, y, w, h]$ normalized between 0 and 1. Specifies the exact spatial location and extent of an object.
2. **IoU (Intersection over Union):**
   Measures how well the predicted bounding box overlaps with the ground truth human annotation:
   $$\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}$$
3. **Precision:**
   Out of all items the model *claimed* were litter, what percentage were actually litter? (High precision = few false alarms).
   $$\text{Precision} = \frac{TP}{TP + FP}$$
4. **Recall:**
   Out of all actual litter items present on the beach, what percentage did the model successfully find? (High recall = few missed items).
   $$\text{Recall} = \frac{TP}{TP + FN}$$
5. **mAP@50 (Mean Average Precision at 0.5 IoU):**
   The standard benchmark metric. Considers a detection correct if its bounding box overlaps the ground truth by at least 50% ($0.50$).
6. **mAP@50–95:**
   The strictest benchmark metric. Averages the mAP calculated across 10 IoU thresholds from 0.50 to 0.95 in steps of 0.05.
