# PanScope: Intelligent PAN Card Data Extractor 🧾🔍

**PanScope** is a smart document parsing system that extracts structured information from Indian PAN cards. It combines OCR, deep learning-based segmentation, and NLP-driven metadata extraction to accurately retrieve key fields from both old and new PAN card formats.

## ✨ Features

- 🧠 OCR + Deep Learning for accurate text localization
- 🔤 Character detection and recognition pipelines
- 📄 Supports **both old and new PAN card layouts**
- 🧬 NLP-based metadata parsing for:
  - Name
  - Father's Name
  - Date of Birth
  - PAN Number
- 📊 Evaluation support with metrics: precision, recall, F1-score, mean IoU
- 🖼️ Visualization of detection results with bounding boxes
- 📁 Batch mode for evaluating entire datasets

## 📂 File Overview

| File          | Description |
|---------------|-------------|
| `mm.py`       | Classic demo: run end-to-end extraction on a single image |
| `m3.py`       | Visual mode: overlay predicted vs. ground-truth bounding boxes |
| `m2.py`       | Batch mode: evaluate accuracy on an entire dataset of PAN card images |

## 🏗️ Architecture

- **OCR Engine**: Uses [doctr](https://github.com/mindee/doctr) with selectable backends (PyTorch/TensorFlow)
- **Detection & Recognition**: Heatmap segmentation + bounding box extraction
- **Text Analysis**: Regex + geometric heuristics to infer layout
- **NLP Extraction**: Extracts and labels PAN Number, DOB, Name, and Father's Name


## 📊 Metrics Reported

- **IoU** (Intersection over Union)
- **Precision**
- **Recall**
- **F1 Score**
- **Accuracy** (based on matched boxes)

## 🧪 Sample Output

```json
{
  "layout": "new",
  "Name": "",
  "Father's Name": "",
  "Date of Birth": "",
  "PAN Number": ""
}
```
🚀 How to Run
1. Install Dependencies
```bash
pip install -r requirements.txt
```
2. Run Classic Demo
```bash
python mm.py
```
3. Visual Evaluation (Single Image)
```bash
python m3.py
```
4. Batch Evaluation on Dataset
```bash
python m2.py
```
Make sure to place your dataset under Pan Cards.v1i.yolov12/train/images and corresponding YOLO labels in train/labels.

🔒 Requirements
Python 3.8+

doctr (OCR engine)

OpenCV, NumPy, Matplotlib

PyTorch or TensorFlow

Scikit-learn

📃 License
MIT License
