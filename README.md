# 🧠 AlzheimerAI — Alzheimer's Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/TensorFlow-2.13%2B-orange?style=for-the-badge&logo=tensorflow" />
  <img src="https://img.shields.io/badge/Flask-3.0-green?style=for-the-badge&logo=flask" />
  <img src="https://img.shields.io/badge/scikit--learn-1.3%2B-yellow?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge" />
</p>

<p align="center">
  An AI-powered <strong>dual-modality</strong> Alzheimer's detection system combining<br/>
  <strong>clinical biomarker analysis</strong> (ML) and <strong>MRI brain scan classification</strong> (CNN).
</p>

---

## 📌 Overview

AlzheimerAI is a full-stack deep learning web application designed to assist in the **early detection of Alzheimer's disease**. It supports two complementary detection pipelines:

- **Machine Learning Pipeline** — Analyzes 33 clinical features (age, cognitive scores, biomarkers) using ensemble models (Random Forest, Gradient Boosting) to predict Alzheimer's risk.
- **CNN Pipeline** — Classifies MRI brain scans into 4 dementia stages using a fine-tuned **EfficientNetB0** convolutional neural network.

> ⚠️ **Disclaimer:** This system is for **research and educational purposes only**. It is NOT a certified medical device. Always consult a qualified neurologist for clinical diagnosis.

---

## 🎯 Key Features

| Feature | Description |
|---|---|
| 🔬 **ML Detection** | 33 clinical features → Binary Alzheimer's classification |
| 🧬 **CNN Detection** | MRI image → 4-class dementia staging |
| 📊 **Dashboard** | Interactive training graphs with zoom support |
| ⚖️ **Risk Assessment** | Low / Medium / High risk level output |
| 📈 **Feature Importance** | Top contributing biomarkers visualized |
| 🎯 **Confidence Scores** | Per-class probability bars |
| 🖼️ **Grad-CAM** | Visual explanation of CNN predictions |
| 🌐 **Responsive UI** | Dark theme, mobile-friendly Flask app |

---

## 📊 Model Performance

| Model | Metric | Score |
|---|---|---|
| Random Forest (ML) | Test Accuracy | **94.42%** |
| Random Forest (ML) | ROC-AUC | **94.09%** |
| Gradient Boosting (ML) | CV Accuracy | **94.88%** |
| EfficientNetB0 (CNN) | Test Accuracy | *Depends on dataset size* |

---

## 🏥 CNN Classification Classes (MRI)

| Class | Description |
|---|---|
| 🟢 **Non Demented** | No significant signs of cognitive decline |
| 🟡 **Very Mild Demented** | Subtle memory lapses, earliest stage |
| 🟠 **Mild Demented** | Noticeable memory and functional impairment |
| 🔴 **Moderate Demented** | Significant cognitive and functional decline |

---

## 📁 Project Structure

```
Alzheimer-s-Detection/
├── app.py                  ← Flask web application & API routes
├── train_ml.py             ← ML training (Random Forest, CSV data)
├── train_cnn.py            ← CNN training (EfficientNetB0, MRI images)
├── balance_dataset.py      ← Dataset balancing / oversampling
├── alzheimer_dataset.csv   ← Clinical biomarker dataset
├── requirements.txt        ← Python dependencies
│
├── models/                 ← Saved trained model artifacts
│   ├── ml_model.pkl        ← Trained ML classifier
│   ├── ml_scaler.pkl       ← Feature scaler
│   ├── ml_artifacts.pkl    ← Label encoders & metadata
│   ├── cnn_model.keras     ← Trained CNN model
│   └── cnn_artifacts.pkl   ← CNN class labels & metadata
│
├── graphs/                 ← Auto-generated training graphs (PNG)
│
├── static/
│   ├── graphs/             ← Graphs served by Flask
│   └── uploads/            ← Uploaded MRI images (temp)
│
├── templates/              ← Jinja2 HTML templates
│   ├── base.html
│   ├── index.html
│   ├── ml_predict.html
│   ├── ml_result.html
│   ├── cnn_predict.html
│   ├── cnn_result.html
│   ├── dashboard.html
│   └── about.html
│
└── dataset/                ← MRI dataset (you must provide)
    ├── train/
    │   ├── Non_Demented/
    │   ├── Very_Mild_Demented/
    │   ├── Mild_Demented/
    │   └── Moderate_Demented/
    └── test/
        └── ...
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager
- *(Optional)* GPU with CUDA support for faster CNN training

### 1. Clone the Repository

```bash
git clone https://github.com/Hannath-anna/Alzheimer-s-Detection.git
cd Alzheimer-s-Detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the ML Model (Clinical Data)

```bash
python train_ml.py
```

Trains a Random Forest classifier on `alzheimer_dataset.csv`. Outputs:
- `models/ml_model.pkl`, `models/ml_scaler.pkl`, `models/ml_artifacts.pkl`
- `graphs/ml_*.png` — 8 training visualization graphs

### 4. Train the CNN Model (MRI Scans)

First, download and organize the MRI dataset:

> 📦 Recommended dataset: [Alzheimer's Dataset (4 classes) on Kaggle](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images)

Place images as:
```
dataset/train/{Non_Demented, Very_Mild_Demented, Mild_Demented, Moderate_Demented}/
dataset/test/{...same structure...}
```

Then run:
```bash
python train_cnn.py
```

Outputs:
- `models/cnn_model.keras`
- `models/cnn_artifacts.pkl`
- `graphs/cnn_*.png` — 5 training graphs

### 5. Copy Graphs to Static Folder

```bash
# Linux/macOS
cp graphs/*.png static/graphs/

# Windows
copy graphs\*.png static\graphs\
```

### 6. Launch the Web App

```bash
python app.py
```

Open your browser at: **[http://localhost:5000](http://localhost:5000)**

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Backend** | Python 3.10, Flask 3.0 |
| **ML Models** | scikit-learn, XGBoost |
| **Deep Learning** | TensorFlow 2.13, Keras (EfficientNetB0) |
| **Data Processing** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Image Handling** | Pillow (PIL) |
| **Frontend** | HTML5, CSS3, Jinja2 |

---

## 🔬 Architecture Overview

```
             ┌─────────────────────────────┐
             │        Flask Web App        │
             │           app.py            │
             └───────────┬─────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
   ┌──────▼──────┐               ┌──────▼──────┐
   │  ML Pipeline │               │ CNN Pipeline │
   │ train_ml.py  │               │ train_cnn.py │
   └──────┬──────┘               └──────┬───────┘
          │                             │
   ┌──────▼──────┐               ┌──────▼───────┐
   │  CSV Dataset │               │  MRI Images  │
   │ (33 features)│               │ (4 classes)  │
   └──────┬──────┘               └──────┬───────┘
          │                             │
   ┌──────▼──────┐               ┌──────▼───────┐
   │Random Forest │               │EfficientNetB0│
   │+ Grad Boost  │               │ (Fine-tuned) │
   └─────────────┘               └──────────────┘
```

---

## 📝 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 👩‍💻 Author

**Hannath Anna** — [@Hannath-anna](https://github.com/Hannath-anna)

> Built with ❤️ for early Alzheimer's detection research.
