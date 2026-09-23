import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn

ROOT = Path(__file__).parent
NUM_COLS = ["age", "hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level"]
CAT_COLS = ["gender", "smoking_history"]

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="wide")
st.markdown(
    """<style>
    [data-testid="stMetric"]{background:rgba(128,128,128,.08);border:1px solid rgba(128,128,128,.2);
    padding:12px 16px;border-radius:12px}
    </style>""",
    unsafe_allow_html=True,
)


# ---------- Loading ----------
@st.cache_resource
def load_artifacts():
    # Same architecture as the training notebook
    model = nn.Sequential(
        nn.Linear(15, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, 8), nn.ReLU(),
        nn.Dropout(0.5), nn.Linear(8, 1), nn.Sigmoid(),
    )
    model.load_state_dict(torch.load(ROOT / "artifacts" / "model_diabet.pt", map_location="cpu"))
    model.eval()
    with open(ROOT / "artifacts" / "OneHotEncoder.pkl", "rb") as f:
        ohe = pickle.load(f)
    return model, ohe


@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "Dataset" / "diabetes_prediction_dataset.csv")


model, ohe = load_artifacts()
data = load_data()


def predict(df: pd.DataFrame) -> np.ndarray:
    """Returns diabetes probability for each row.
    NOTE: no StandardScaler here on purpose. In the training notebook the scaler's output
    was never assigned back to x, so the model was trained on UNSCALED features."""
    x = np.hstack([df[NUM_COLS].to_numpy(float), ohe.transform(df[CAT_COLS])])
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).squeeze(1).numpy()


# ---------- Sidebar ----------
with st.sidebar:
    st.header("🧑‍⚕️ Patient details")
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    age = st.slider("Age", 1, 80, 40)
    bmi = st.slider("BMI", 10.0, 70.0, 27.0, 0.1)
    hba1c = st.slider("HbA1c level (%)", 3.5, 9.0, 5.7, 0.1)
    glucose = st.slider("Blood glucose (mg/dL)", 80, 300, 120)
    smoking = st.selectbox("Smoking history", ["never", "No Info", "current", "former", "ever", "not current"])
    hypertension = st.toggle("Hypertension")
    heart_disease = st.toggle("Heart disease")
    st.divider()
    threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05,
                          help="Probability above which the patient is flagged as high risk.")

patient = pd.DataFrame([{
    "gender": gender, "age": age, "hypertension": int(hypertension), "heart_disease": int(heart_disease),
    "smoking_history": smoking, "bmi": bmi, "HbA1c_level": hba1c, "blood_glucose_level": glucose,
}])

# ---------- Header ----------
st.title("🩺 Diabetes Risk Predictor")
st.caption("A PyTorch neural network trained on 100,000 patient records.")

tab_pred, tab_batch, tab_data, tab_about = st.tabs(["🔮 Prediction", "📁 Batch", "📊 Data insights", "ℹ️ About"])

# ---------- Single prediction ----------
with tab_pred:
    prob = float(predict(patient)[0])
    positive = prob >= threshold
    color = "#e74c3c" if positive else "#2ecc71"

    left, right = st.columns([1.2, 1])
    with left:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100, number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Diabetes probability"},
            gauge={
                "axis": {"range": [0, 100]}, "bar": {"color": color},
                "steps": [{"range": [0, threshold * 100], "color": "rgba(46,204,113,.15)"},
                          {"range": [threshold * 100, 100], "color": "rgba(231,76,60,.15)"}],
                "threshold": {"line": {"color": "gray", "width": 3}, "value": threshold * 100},
            },
        ))
        fig.update_layout(height=320, margin=dict(t=60, b=10, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)
        if positive:
            st.error("**High risk** – the model flags this patient as likely diabetic.")
        else:
            st.success("**Low risk** – the model does not flag this patient as diabetic.")

    with right:
        st.subheader("Compared to dataset average")
        avg = data[NUM_COLS].mean()
        c1, c2 = st.columns(2)
        c1.metric("Age", f"{age}", f"{age - avg['age']:+.1f}", delta_color="off")
        c2.metric("BMI", f"{bmi:.1f}", f"{bmi - avg['bmi']:+.1f}", delta_color="off")
        c1.metric("HbA1c", f"{hba1c:.1f} %", f"{hba1c - avg['HbA1c_level']:+.1f}", delta_color="off")
        c2.metric("Glucose", f"{glucose}", f"{glucose - avg['blood_glucose_level']:+.0f}", delta_color="off")

    st.warning("This tool is for educational purposes only and is not a medical diagnosis.", icon="⚠️")

# ---------- Batch prediction ----------
with tab_batch:
    st.write(f"Upload a CSV with these columns: `{', '.join(CAT_COLS + NUM_COLS)}`")
    file = st.file_uploader("CSV file", type="csv")
    if file:
        try:
            batch = pd.read_csv(file)
            batch["diabetes_probability"] = predict(batch).round(4)
            batch["prediction"] = np.where(batch["diabetes_probability"] >= threshold, "Diabetic", "Not diabetic")
            st.dataframe(batch, use_container_width=True)
            st.download_button("⬇️ Download results", batch.to_csv(index=False), "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not process the file: {e}")

# ---------- Data insights ----------
with tab_data:
    m1, m2, m3 = st.columns(3)
    m1.metric("Records", f"{len(data):,}")
    m2.metric("Diabetic patients", f"{data['diabetes'].mean():.1%}")
    m3.metric("Features", len(NUM_COLS + CAT_COLS))

    col_a, col_b = st.columns(2)
    with col_a:
        counts = data["diabetes"].map({0: "No diabetes", 1: "Diabetes"}).value_counts()
        st.plotly_chart(px.pie(values=counts.values, names=counts.index, hole=0.5, title="Class distribution"),
                        use_container_width=True)
    with col_b:
        corr = data[NUM_COLS + ["diabetes"]].corr()
        st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Correlation"),
                        use_container_width=True)

    feature = st.selectbox("Feature distribution", ["HbA1c_level", "blood_glucose_level", "age", "bmi"])
    hist = px.histogram(data.assign(diabetes=data["diabetes"].map({0: "No", 1: "Yes"})), x=feature,
                        color="diabetes", barmode="overlay", opacity=0.65, nbins=50)
    hist.add_vline(x=patient[feature].iloc[0], line_dash="dash", annotation_text="Patient")
    st.plotly_chart(hist, use_container_width=True)

# ---------- About ----------
with tab_about:
    st.markdown(
        """
**Model:** fully-connected network `15 → 64 → 32 → 16 → 8 → 1` (ReLU, Dropout 0.5, Sigmoid), trained with BCE loss and Adam.

**Preprocessing:** one-hot encoding for gender and smoking history, SMOTE to balance classes during training.

**Limitations**
- Training data was balanced with SMOTE, so probabilities are not calibrated to real-world prevalence (about 8.5% in this dataset).
- The model only saw ages 0–80 and glucose values 80–300 mg/dL.
- Not a substitute for clinical testing or a doctor's advice.
        """
    )
