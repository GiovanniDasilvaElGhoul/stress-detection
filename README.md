# 🧠 Stress Detection Using 1D CNN on WESAD Dataset

A deep learning project that uses a **1D Convolutional Neural Network (CNN)** to classify whether a person is **stressed or not** based on physiological signals from the WESAD dataset.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)

---

## 📖 Overview

Stress is a major health concern in modern society. This project aims to automatically detect stress using **wearable physiological signals** such as ECG, EDA, EMG, Respiration, and Temperature — without relying on self-reporting or subjective measures.

The model is trained and evaluated using **Leave-One-Subject-Out (LOSO) Cross-Validation**, which is the standard evaluation method for subject-independent stress detection.

---

## 📂 Dataset

**WESAD (Wearable Stress and Affect Detection)**

| Property | Details |
|---|---|
| Subjects | 15 subjects (S2–S17, S1 and S12 excluded) |
| Device (Chest) | RespiBAN @ 700 Hz |
| Device (Wrist) | Empatica E4 @ 32–64 Hz |
| Labels | 0: Undefined, 1: Baseline, 2: Stress, 3: Amusement, 4: Meditation |
| Used Labels | 1 (Baseline) and 2 (Stress) only |

### Signals Used (Chest)
| Signal | Description | Sampling Rate |
|---|---|---|
| ECG | Electrocardiogram | 700 Hz |
| EDA | Electrodermal Activity | 700 Hz |
| EMG | Electromyogram | 700 Hz |
| Resp | Respiration | 700 Hz |
| Temp | Body Temperature | 700 Hz |

> 📥 Dataset available at: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/

---

## 🔄 Project Pipeline

```
Raw WESAD .pkl files
        ↓
🧹 Signal Filtering (Bandpass / Lowpass)
        ↓
✂️  Sliding Window Segmentation (1 sec, 50% overlap)
        ↓
🏷️  Binary Label Encoding (0=Baseline, 1=Stress)
        ↓
📏 Normalization (StandardScaler)
        ↓
🧠 1D CNN Model Training
        ↓
📊 LOSO Cross-Validation Evaluation
        ↓
📈 Accuracy / F1 Score / Results Visualization
```

---

## 🧠 Model Architecture

A **1D Convolutional Neural Network** designed for time-series physiological signals:

```
Input → (700, 5)    [1 second window × 5 signals]
        ↓
Conv1D(64, kernel=5) + BatchNorm + MaxPool + Dropout(0.3)
        ↓
Conv1D(128, kernel=3) + BatchNorm + MaxPool + Dropout(0.3)
        ↓
Conv1D(256, kernel=3) + BatchNorm + GlobalAvgPool + Dropout(0.4)
        ↓
Dense(128) + Dropout(0.5)
        ↓
Dense(64)
        ↓
Dense(1, sigmoid)   [0=Baseline, 1=Stress]
```

| Layer | Purpose |
|---|---|
| Conv1D | Extract temporal patterns from signals |
| BatchNormalization | Stabilize and speed up training |
| MaxPooling1D | Reduce dimensionality |
| GlobalAveragePooling1D | Summarize learned features |
| Dropout | Prevent overfitting |
| Dense + Sigmoid | Binary stress classification |

---

## 📊 Results

Evaluated using **Leave-One-Subject-Out (LOSO) Cross-Validation** across all 15 subjects.

| Metric | Score |
|---|---|
| Average Accuracy | TBD after training |
| Average F1 Score | TBD after training |
| Std Accuracy | TBD after training |

> Results will be updated after full training is complete.

---

## ⚙️ Installation

### Option 1: Google Colab (Recommended)
```python
# Install required packages
!pip install scipy scikit-learn tqdm

# Download dataset from Kaggle
import os
os.environ['KAGGLE_USERNAME'] = 'your_username'
os.environ['KAGGLE_KEY']      = 'your_api_key'
!kaggle datasets download -d nguynngcthuanh/wesad-dataset --path /content/
!unzip -o /content/wesad-dataset.zip -d /content/WESAD/
```

### Option 2: Local (PyCharm)
```bash
pip install numpy pandas scipy scikit-learn tensorflow matplotlib seaborn tqdm
```

---

## 🚀 Usage

### Step 1: Import Libraries
```python
import os, pickle
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, BatchNormalization, GlobalAveragePooling1D
```

### Step 2: Set Dataset Path
```python
DATASET_PATH = '/content/WESAD/WESAD/'
SUBJECT_IDS  = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
```

### Step 3: Preprocess & Train
Run the preprocessing cells to filter, segment, and normalize signals, then train the CNN using LOSO cross-validation.

### Step 4: Evaluate
```python
print(f"Average Accuracy : {np.mean(all_accuracies):.4f}")
print(f"Average F1 Score : {np.mean(all_f1_scores):.4f}")
```

---

## 📁 Project Structure

```
stress-detection-cnn/
│
├── README.md                   ← This file
│
├── stress_detection.ipynb      ← Main Google Colab notebook
│
├── data/
│   ├── X_scaled.npy            ← Preprocessed signals (saved)
│   ├── y.npy                   ← Labels (saved)
│   └── groups.npy              ← Subject IDs (saved)
│
└── results/
    ├── accuracy_per_subject.png
    └── f1_per_subject.png
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.x | Programming language |
| TensorFlow / Keras | Building and training the CNN |
| NumPy | Numerical computations |
| SciPy | Signal filtering |
| Scikit-learn | Preprocessing and evaluation |
| Matplotlib / Seaborn | Visualization |
| Google Colab | Cloud training environment |
| Kaggle | Dataset hosting |

---

## 📚 References

- Schmidt, P., et al. (2018). **Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection**. ICMI 2018.
- WESAD Dataset: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/

---

## 👤 Author

> Project developed as part of a deep learning course on physiological signal classification.

---

## 📄 License

This project is for academic and research purposes only. The WESAD dataset is subject to its own terms of use.
