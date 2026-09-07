import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
)

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

# Columns where 0 is not a real value in this dataset -> treated as missing
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

MEDIANS = {
    "Glucose": 117.0,
    "BloodPressure": 72.0,
    "SkinThickness": 29.0,
    "Insulin": 125.0,
    "BMI": 32.3,
}


# ----------------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("artifacts/model_diabet.keras")
    with open("artifacts/PCA.pkl", "rb") as f:
        pca = pickle.load(f)
    with open("artifacts/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, pca, scaler


@st.cache_data
def load_dataset():
    df = pd.read_csv("Dataset/diabetes_diabet.csv")
    return df


model, pca, scaler = load_artifacts()
df_raw = load_dataset()


def predict(raw_values: dict) -> float:
    """raw_values: dict of the 8 raw feature values -> returns probability of diabetes."""
    row = pd.DataFrame([raw_values])[FEATURES]
    for col in ZERO_AS_MISSING:
        if row.loc[0, col] == 0:
            row.loc[0, col] = MEDIANS[col]
    X = row.values
    X = pca.transform(X)
    X = scaler.transform(X)
    prob = model.predict(X, verbose=0)[0][0]
    return float(prob)


# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
st.sidebar.title("🩺 Diabetes Predictor")
page = st.sidebar.radio("Go to", ["Predict", "Batch Prediction", "Data Explorer", "About"])

# ----------------------------------------------------------------------------
# PAGE: Predict
# ----------------------------------------------------------------------------
if page == "Predict":
    st.title("Diabetes Risk Prediction")
    st.caption("Enter patient measurements to estimate diabetes risk. Model: Keras neural network (PCA + StandardScaler pipeline).")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=117)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=72)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=23)

    with col2:
        insulin = st.number_input("Insulin (mu U/mL)", min_value=0, max_value=900, value=30)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=32.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.37, step=0.01)
        age = st.number_input("Age", min_value=1, max_value=120, value=30)

    st.caption("Tip: leave Glucose, Blood Pressure, Skin Thickness, Insulin or BMI at 0 if unknown — the app will fill them with the dataset median.")

    if st.button("Predict", type="primary", use_container_width=True):
        raw = {
            "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness, "Insulin": insulin, "BMI": bmi,
            "DiabetesPedigreeFunction": dpf, "Age": age,
        }
        prob = predict(raw)
        pred = "Diabetes" if prob > 0.5 else "No Diabetes"

        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1:
            if pred == "Diabetes":
                st.error(f"### Result: {pred}")
            else:
                st.success(f"### Result: {pred}")
            st.metric("Predicted probability of diabetes", f"{prob*100:.1f}%")

        with c2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#d62728" if prob > 0.5 else "#2ca02c"},
                    "steps": [
                        {"range": [0, 50], "color": "#eafaf1"},
                        {"range": [50, 100], "color": "#fdecea"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "value": 50},
                },
                title={"text": "Diabetes Risk"},
            ))
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.caption("⚠️ This is an educational demo, not a medical diagnosis. Consult a healthcare professional for real assessment.")

# ----------------------------------------------------------------------------
# PAGE: Batch Prediction
# ----------------------------------------------------------------------------
elif page == "Batch Prediction":
    st.title("Batch Prediction")
    st.caption(f"Upload a CSV with columns: {', '.join(FEATURES)}")

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        batch_df = pd.read_csv(file)
        missing_cols = set(FEATURES) - set(batch_df.columns)
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
        else:
            probs = []
            for _, row in batch_df[FEATURES].iterrows():
                probs.append(predict(row.to_dict()))
            batch_df["Diabetes_Probability"] = np.round(probs, 4)
            batch_df["Prediction"] = np.where(batch_df["Diabetes_Probability"] > 0.5, "Diabetes", "No Diabetes")

            st.dataframe(batch_df, use_container_width=True)
            st.download_button(
                "Download results as CSV",
                batch_df.to_csv(index=False).encode("utf-8"),
                "predictions.csv",
                "text/csv",
            )

            fig = px.histogram(batch_df, x="Diabetes_Probability", nbins=20,
                                title="Distribution of predicted probabilities")
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# PAGE: Data Explorer
# ----------------------------------------------------------------------------
elif page == "Data Explorer":
    st.title("Dataset Explorer")
    st.caption("Pima Indians Diabetes dataset (768 rows)")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total records", len(df_raw))
    c2.metric("Diabetes cases", int(df_raw["Outcome"].sum()))
    c3.metric("Diabetes rate", f"{df_raw['Outcome'].mean()*100:.1f}%")

    st.subheader("Feature distribution by outcome")
    feature = st.selectbox("Choose a feature", FEATURES)
    fig = px.histogram(df_raw, x=feature, color="Outcome", barmode="overlay",
                        color_discrete_map={0: "#2ca02c", 1: "#d62728"},
                        labels={"Outcome": "Outcome"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation heatmap")
    corr = df_raw.corr(numeric_only=True)
    fig2 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Raw data")
    st.dataframe(df_raw, use_container_width=True)

# ----------------------------------------------------------------------------
# PAGE: About
# ----------------------------------------------------------------------------
else:
    st.title("About this app")
    st.markdown("""
    **Pipeline**
    1. Raw features → zero values in `Glucose, BloodPressure, SkinThickness, Insulin, BMI` are treated as missing and filled with the training-set median.
    2. `PCA` (4 components) reduces the 8 features.
    3. `StandardScaler` normalizes the PCA output.
    4. A Keras neural network (Dense 32 → Dense 16 → Dropout → sigmoid) predicts diabetes probability.

    **Files used**
    - `model_diabet.keras` — trained Keras model
    - `PCA.pkl` — fitted PCA transformer
    - `scaler_diabet.pkl` — fitted StandardScaler
    - `diabetes_diabet.csv` — dataset used for the Data Explorer tab

    Built with Streamlit, TensorFlow/Keras, and scikit-learn.
    """)
