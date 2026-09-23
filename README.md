# 🩺 Diabetes Risk Predictor

An end-to-end machine learning project that predicts a patient's diabetes risk from basic clinical data. A **PyTorch** neural network is trained on 100,000 patient records and served through an interactive **Streamlit** dashboard with single-patient prediction, batch scoring, and data exploration.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

<!-- Add a screenshot or GIF of the app here:
![App demo](assets/demo.png)
-->

---

## 📌 Table of Contents

- [Features](#-features)
- [Dataset](#-dataset)
- [Model & Methodology](#-model--methodology)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Batch Prediction Format](#-batch-prediction-format)
- [Limitations & Future Work](#-limitations--future-work)
- [Disclaimer](#-disclaimer)
- [License](#-license)

---

## ✨ Features

The Streamlit app is organized into four tabs:

| Tab | Description |
|-----|-------------|
| 🔮 **Prediction** | Enter patient details in the sidebar and get an instant diabetes probability on a gauge chart, with an adjustable decision threshold and a comparison against dataset averages. |
| 📁 **Batch** | Upload a CSV file, get a probability and a label for every row, and download the results. |
| 📊 **Data Insights** | Class distribution, correlation heatmap, and interactive feature histograms with the current patient marked on the plot. |
| ℹ️ **About** | Model summary, preprocessing steps, and known limitations. |

## 📊 Dataset

The project uses the **Diabetes Prediction Dataset** (`Dataset/diabetes_prediction_dataset.csv`): **100,000 records**, of which about **8.5 %** are diabetic (8,500 positive vs. 91,500 negative), so the classes are heavily imbalanced.

| Feature | Type | Description |
|---------|------|-------------|
| `gender` | Categorical | Female, Male, Other |
| `age` | Numeric | Age in years |
| `hypertension` | Binary | 0 = no, 1 = yes |
| `heart_disease` | Binary | 0 = no, 1 = yes |
| `smoking_history` | Categorical | never, No Info, current, former, ever, not current |
| `bmi` | Numeric | Body mass index |
| `HbA1c_level` | Numeric | Average blood sugar level over the past 2–3 months (%) |
| `blood_glucose_level` | Numeric | Blood glucose (mg/dL) |
| `diabetes` | Binary | **Target**: 0 = no diabetes, 1 = diabetes |

A Gradient Boosting feature-importance check in the notebook shows that **HbA1c level** and **blood glucose level** are by far the most influential features, followed by age and hypertension.

## 🧠 Model & Methodology

**Pipeline**

1. **Encoding**: `OneHotEncoder` on `gender` and `smoking_history` → 6 numeric + 9 one-hot columns = **15 input features**.
2. **Class balancing**: `SMOTE` (`random_state=42`) to oversample the minority class.
3. **Split**: 80 % train / 10 % validation / 10 % test.
4. **Training**: mini-batch training with early stopping on validation loss; the best weights are restored at the end.

**Architecture**

```
Input (15) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(16) → ReLU
           → Linear(8)  → ReLU → Dropout(0.5) → Linear(1) → Sigmoid
```

**Training configuration**

| Setting | Value |
|---------|-------|
| Loss | Binary Cross-Entropy (`BCELoss`) |
| Optimizer | Adam, learning rate `0.001` |
| Batch size | 100 |
| Max epochs | 100 |
| Early stopping | Patience 10 on validation loss (stopped at epoch 31) |

## 🏆 Results

Metrics on the held-out test split (18,300 samples, decision threshold 0.5):

| Metric | Score |
|--------|-------|
| Accuracy | **0.9056** |
| Precision | 0.8835 |
| Recall | **0.9346** |
| F1-score | 0.9083 |

**Confusion matrix**

|  | Predicted: No Diabetes | Predicted: Diabetes |
|--|:--:|:--:|
| **Actual: No Diabetes** | 8,010 | 1,129 |
| **Actual: Diabetes** | 599 | 8,562 |

> ⚠️ These numbers are measured on a **SMOTE-balanced** dataset (see [Limitations](#-limitations--future-work)), so they should not be read as real-world performance.

## 📁 Project Structure

```
Diabetes-Predict/
├── Dataset/
│   └── diabetes_prediction_dataset.csv   # Raw data (100k rows)
├── artifacts/
│   ├── model_diabet.pt                   # Trained PyTorch weights (state_dict)
│   └── OneHotEncoder.pkl                 # Fitted encoder used at inference
├── Notebook_Project.ipynb                # EDA, preprocessing, training, evaluation
├── app.py                                # Streamlit application
├── requirements.txt                      # App dependencies
├── LICENSE
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Diabetes-Predict.git
cd Diabetes-Predict
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Reproduce the training (optional)

The notebook needs a few extra packages:

```bash
pip install imbalanced-learn matplotlib jupyter
jupyter notebook Notebook_Project.ipynb
```

Make sure the CSV path in the first cells points to `Dataset/diabetes_prediction_dataset.csv`.

## 📥 Batch Prediction Format

Upload a CSV to the **Batch** tab with these columns:

```
gender, smoking_history, age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level
```

Example:

```csv
gender,smoking_history,age,hypertension,heart_disease,bmi,HbA1c_level,blood_glucose_level
Female,never,54,0,0,27.3,6.6,140
Male,current,67,1,1,31.8,7.2,210
```

The app returns two extra columns, `diabetes_probability` and `prediction`, which you can download as `predictions.csv`.

## ⚠️ Limitations & Future Work

Being transparent about where this project can be improved:

- **SMOTE before splitting.** Oversampling is applied to the full dataset before the train/validation/test split, so synthetic samples derived from the same real patients can appear in both training and test data. This likely makes the reported metrics optimistic. *Fix:* split first, apply SMOTE to the training set only, and evaluate on the untouched, imbalanced test set.
- **No feature scaling.** A `StandardScaler` was fitted in the notebook but its output was never used, so the network is trained (and served) on raw feature values. Adding scaling would probably improve convergence.
- **Uncalibrated probabilities.** Because training data was balanced, the output probability is not calibrated to the real ~8.5 % prevalence.
- **Unseeded split.** The train/validation/test split has no `random_state`, so exact numbers vary between runs.
- **Limited input range.** The model has only seen ages 0–80 and glucose values 80–300 mg/dL.
- **Better evaluation.** Report ROC-AUC, PR-AUC and threshold analysis, and compare against strong tabular baselines such as XGBoost or LightGBM.

## 🩺 Disclaimer

This project is for **educational and research purposes only**. It is **not** a medical device and must not be used for diagnosis or treatment decisions. Always consult a qualified healthcare professional.

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

Copyright © 2026 Mahan Liaghatmand
