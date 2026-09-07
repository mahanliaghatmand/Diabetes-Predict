# 🩺 Diabetes Predict

**Diabetes risk prediction using a Keras neural network trained on the Pima Indians Diabetes dataset, deployed as an interactive Streamlit app for single prediction, batch prediction, and data exploration.**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)
![Scikit--learn](https://img.shields.io/badge/scikit--learn-ML-f89939.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 Overview

This project solves a **binary classification** problem in the **healthcare** domain: given 8 clinical measurements for a patient, it predicts the probability of diabetes.

The project has two main components:

1. **Data Science / Modeling** (`diabetes_classification.ipynb`) — data cleaning, feature engineering, dimensionality reduction with PCA, feature scaling, neural network design and training, and full evaluation.
2. **Deployment** (`app.py`) — a multi-page Streamlit application that serves the trained model for real, interactive use.

> ⚠️ **Disclaimer:** This project is for educational purposes and to demonstrate machine learning skills. It is **not a substitute for medical diagnosis**.

---

## 🗂️ Dataset

| Property | Value |
|---|---|
| Name | Pima Indians Diabetes Dataset |
| Samples | 768 |
| Features | 8 input features + 1 target (`Outcome`) |
| Class distribution | 500 samples of `0` (no diabetes) / 268 samples of `1` (diabetes) — imbalanced (~65% vs ~35%) |

**Input features:**

`Pregnancies` · `Glucose` · `BloodPressure` · `SkinThickness` · `Insulin` · `BMI` · `DiabetesPedigreeFunction` · `Age`

---

## 🔬 Data Preprocessing Pipeline

1. **Handling missing values:**
   In this dataset, a value of `0` in the columns `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` is physiologically impossible and actually represents missing data. These values are first converted to `NaN` and then imputed with the **median** of the respective column.

2. **Dimensionality reduction with PCA:**
   The 8 input features are reduced to 4 principal components using `PCA(n_components=4)`, reducing noise and controlling model complexity.

3. **Feature scaling:**
   The PCA output is scaled with `StandardScaler` to have zero mean and unit variance (a standard requirement for stable neural network convergence).

4. **Train/test split:**
   An 80/20 split with `stratify=y` to preserve class balance, and `random_state=42` for reproducibility.

---

## 🧠 Model Architecture

A **Feed-Forward Neural Network (MLP)** built with Keras:

```
Input(4)  →  Dense(32, relu)  →  Dense(16, relu)  →  Dropout(0.2)  →  Dense(1, sigmoid)
```

| Item | Value |
|---|---|
| Total parameters | 705 |
| Optimizer | Adam (`learning_rate=0.001`) |
| Loss function | Binary Crossentropy |
| Training metric | Accuracy |
| Callbacks | `EarlyStopping` (patience=30, restores best weights) + `ReduceLROnPlateau` |
| Epochs | Up to 200 (with early stopping) |

---

## 📊 Results & Evaluation

Evaluated on the held-out test set (154 samples):

| Metric | Value |
|---|---|
| Accuracy | 70.1% |
| F1-score (Diabetes class) | 0.54 |
| Precision (weighted) | 0.69 |
| Recall (weighted) | 0.70 |

**Classification report:**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| No Diabetes | 0.75 | 0.81 | 0.78 | 100 |
| Diabetes | 0.59 | 0.50 | 0.54 | 54 |

**Confusion matrix:**

```
                Pred: No     Pred: Yes
Actual: No        81            19
Actual: Yes       27            27
```

**Analysis:** Due to the class imbalance in the dataset, the model performs noticeably worse on the minority class (`Diabetes`), reflected in its lower recall for that class. This gap points directly to the improvement directions listed below.

---

## 🖥️ Streamlit App

The application (`app.py`) has 4 pages:

- **🔮 Predict** — manually enter a patient's 8 features and get the predicted diabetes probability, shown with an interactive gauge chart. Zero values in sensitive columns are automatically imputed with the dataset median.
- **📁 Batch Prediction** — upload a CSV of multiple patients, run batch predictions, view a histogram of predicted probabilities, and download the results as CSV.
- **📈 Data Explorer** — overall dataset statistics, per-feature distributions split by outcome, and a correlation heatmap across features.
- **ℹ️ About** — a full explanation of the model pipeline for end users.

---

## 📁 Project Structure

```
Diabetes-Predict-main/
│
├── artifacts/
│   ├── model_diabet.keras     # Trained Keras model
│   ├── PCA.pkl                # Fitted PCA transformer
│   └── scaler.pkl             # Fitted StandardScaler
│
├── Dataset/
│   └── diabetes_diabet.csv    # Raw dataset
│
├── diabetes_classification.ipynb   # Full modeling & evaluation notebook
├── app.py                          # Streamlit application
├── requirements.txt
└── LICENSE
```

---

## ⚙️ Installation & Usage

```bash
# Clone the repository
git clone https://github.com/<username>/Diabetes-Predict.git
cd Diabetes-Predict

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

**Requirements:** `streamlit` · `tensorflow` · `scikit-learn` · `pandas` · `numpy` · `plotly`

---

## 🚀 Future Improvements

- Apply class-balancing techniques (`class_weight`, SMOTE) to improve recall on the diabetes class.
- Benchmark against classical algorithms (Random Forest, XGBoost, Logistic Regression) as baselines.
- Add `SHAP` or `LIME` for model explainability — especially important in a healthcare context.
- Use k-fold cross-validation instead of a single train/test split for a more robust performance estimate.
- Tune the decision threshold instead of a fixed 0.5 cutoff, based on the ROC / Precision-Recall curve.

---

## 🧰 Tech Stack

`Python` · `TensorFlow / Keras` · `scikit-learn` (PCA, StandardScaler) · `Pandas` / `NumPy` · `Streamlit` · `Plotly`

---

## 📄 License

This project is released under the **MIT License** — see [LICENSE](./LICENSE) for details.

---

<p align="center">Built with ❤️ for Machine Learning and Data Science</p>
