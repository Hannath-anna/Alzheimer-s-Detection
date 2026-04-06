# 🧠 AlzheimerAI — Alzheimer Detection System

AI-powered dual-modality Alzheimer's detection using **clinical biomarkers** (ML) and **MRI brain scans** (CNN).

---

## 📁 Project Structure

```
alzheimer_system/
├── app.py                  ← Flask web application
├── train_ml.py             ← ML training (Random Forest, CSV data)
├── train_cnn.py            ← CNN training (EfficientNetB0, MRI images)
├── alzheimer_dataset.csv   ← Clinical dataset (provided)
├── requirements.txt        ← Python dependencies
├── models/                 ← Saved trained models
│   ├── ml_model.pkl
│   ├── ml_scaler.pkl
│   ├── ml_artifacts.pkl
│   ├── cnn_model.keras
│   └── cnn_artifacts.pkl
├── graphs/                 ← Training graphs (auto-generated)
├── static/
│   ├── graphs/             ← Graphs served by Flask
│   └── uploads/            ← Uploaded MRI images
├── dataset/                ← MRI dataset (you must provide)
│   ├── train/
│   │   ├── Non_Demented/
│   │   ├── Very_Mild_Demented/
│   │   ├── Mild_Demented/
│   │   └── Moderate_Demented/
│   └── test/
│       └── ...
└── templates/              ← HTML templates
    ├── base.html
    ├── index.html
    ├── ml_predict.html
    ├── ml_result.html
    ├── cnn_predict.html
    ├── cnn_result.html
    ├── dashboard.html
    └── about.html
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the ML model (Clinical Data)
```bash
python train_ml.py
```
This trains a Random Forest classifier on `alzheimer_dataset.csv` and generates:
- `models/ml_model.pkl`, `models/ml_scaler.pkl`
- `graphs/ml_*.png` (8 training graphs)

### 3. Train the CNN model (MRI Scans) — Optional
```bash
# First organize your MRI dataset:
# dataset/train/{Non_Demented, Very_Mild_Demented, Mild_Demented, Moderate_Demented}/
# dataset/test/...
#
# Recommended: Alzheimer's Dataset from Kaggle
# https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images

python train_cnn.py
```
This trains an EfficientNetB0 CNN and generates:
- `models/cnn_model.keras`
- `graphs/cnn_*.png` (5 training graphs)

### 4. Copy graphs to static folder
```bash
cp graphs/*.png static/graphs/
```

### 5. Run the Flask app
```bash
python app.py
```
Open your browser at: **http://localhost:5000**

---

## 🎯 Features

| Feature | Details |
|---|---|
| **ML Detection** | 33 clinical features → Binary classification |
| **CNN Detection** | MRI image → 4-class dementia staging |
| **Dashboard** | All training graphs with zoom-in |
| **Risk Assessment** | Low / Medium / High risk levels |
| **Feature Importance** | Top contributing biomarkers |
| **Confidence Scores** | Per-class probability bars |
| **Responsive UI** | Dark theme, mobile-friendly |

---

## 📊 Model Performance

| Model | Metric | Score |
|---|---|---|
| Random Forest (ML) | Test Accuracy | **94.42%** |
| Random Forest (ML) | ROC AUC | **94.09%** |
| Gradient Boosting | CV Accuracy | **94.88%** |
| EfficientNetB0 (CNN) | Test Accuracy | *Depends on dataset* |

---

## 🏥 CNN Classes (MRI)

| Class | Description |
|---|---|
| **Non Dementia** | No significant signs of cognitive decline |
| **Very Mild Dementia** | Subtle memory lapses, early stage |
| **Mild Dementia** | Noticeable memory and functional impairment |
| **Moderate Dementia** | Significant cognitive decline |

---

## ⚠️ Disclaimer

This system is **for research and educational purposes only**. It is NOT a certified medical device. 
Always consult a qualified neurologist for medical diagnosis.
