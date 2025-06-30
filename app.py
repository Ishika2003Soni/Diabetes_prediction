# app.py
import json, joblib, shap, numpy as np, pandas as pd, streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ---------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Diabetes Risk Suite", page_icon="🩺", layout="wide")
st.title("🩺 Diabetes Risk Prediction Suite")

# ---------------------------------------------------------------
# Helpers: load artifacts
# ---------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load("best_model.pkl")
    with open("metrics.json") as fp:
        metrics = json.load(fp)
    try:
        explainer = joblib.load("shap_explainer.pkl")
    except Exception:
        explainer = None
    return model, metrics, explainer

model, metrics, explainer = load_artifacts()
feature_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# ---------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Single Prediction", "Batch Prediction", "Dataset Explorer", "Model Metrics"],
    index=0,
)

# ---------------------------------------------------------------
# 1️⃣ SINGLE PREDICTION
# ---------------------------------------------------------------
if page == "Single Prediction":
    st.header("🔍 Single‑Patient Risk Assessment")
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose     = st.slider("Glucose", 0, 200, 120)
        bp          = st.slider("Blood Pressure", 0, 140, 70)
        skin        = st.slider("Skin Thickness (mm)", 0, 100, 20)
    with col2:
        insulin     = st.slider("Insulin (μU/mL)", 0, 900, 79)
        bmi         = st.slider("BMI", 0.0, 70.0, 25.0)
        dpf         = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
        age         = st.slider("Age", 10, 100, 33)

    if st.button("Predict Risk"):
        X = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        prob = model.predict_proba(X)[0][1]
        pred = int(prob >= 0.5)
        if pred == 1:
            st.error(f"⚠️ High risk of diabetes.  (Probability {prob:.1%})")
        else:
            st.success(f"✅ Low risk of diabetes.  (Probability {prob:.1%})")

        # SHAP waterfall explanation
        if explainer:
            st.subheader("Feature Contribution (SHAP - Waterfall)")
            shap_vals = explainer.shap_values(pd.DataFrame(X, columns=feature_names))
            shap.initjs()
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_vals[0],
                    base_values=explainer.expected_value,
                    data=X[0],
                    feature_names=feature_names
                ),
                show=False
            )
            fig = plt.gcf()
            image_path = "shap_waterfall.png"
            fig.savefig(image_path, bbox_inches='tight')
            st.pyplot(fig)

            with open(image_path, "rb") as file:
                st.download_button(
                    label="📥 Download Waterfall Plot (PNG)",
                    data=file,
                    file_name="shap_waterfall.png",
                    mime="image/png"
                )

# ---------------------------------------------------------------
# 2️⃣ BATCH PREDICTION
# ---------------------------------------------------------------
elif page == "Batch Prediction":
    st.header("📄 Batch Prediction from CSV")
    st.write("Upload a **CSV** containing the eight feature columns exactly as in the Pima dataset.")
    file = st.file_uploader("Choose CSV file", type=["csv"])
    if file is not None:
        data = pd.read_csv(file)
        if set(feature_names).issubset(data.columns):
            probs = model.predict_proba(data[feature_names])[:, 1]
            data["Diabetes_Risk(%)"] = (probs * 100).round(1)
            data["Prediction"] = np.where(probs >= 0.5, "High Risk", "Low Risk")
            st.success("Predictions completed:")
            st.dataframe(data.head())
            csv = data.to_csv(index=False).encode()
            st.download_button("⬇️ Download full results", csv, "predictions.csv", "text/csv")
        else:
            st.error("CSV missing required columns!")

# ---------------------------------------------------------------
# 3️⃣ DATASET EXPLORER
# ---------------------------------------------------------------
elif page == "Dataset Explorer":
    st.header("📊 Dataset Explorer")
    df = pd.read_csv("diabetes.csv")
    st.subheader("First 10 rows")
    st.dataframe(df.head(10))

    st.subheader("Summary statistics")
    st.dataframe(df.describe().T)

    st.subheader("Correlation heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, ax=ax)
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Outcome distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="Outcome", data=df, ax=ax)
        st.pyplot(fig)
    with col2:
        st.subheader("Glucose distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["Glucose"], kde=True, ax=ax)
        st.pyplot(fig)

# ---------------------------------------------------------------
# 4️⃣ MODEL METRICS & INTERPRETABILITY
# ---------------------------------------------------------------
elif page == "Model Metrics":
    st.header("📈 Model Comparison & Diagnostics")

    st.subheader("Validation Metrics")
    st.json(metrics)

    st.subheader("Confusion Matrix (best model)")
    df = pd.read_csv("diabetes.csv")
    X = df[feature_names]
    y_true = df["Outcome"]
    y_pred = model.predict(X)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cbar=False)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("ROC Curve (best model)")
    y_prob = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    st.pyplot(fig)

    # SHAP summary plot
    if explainer:
        st.subheader("SHAP Summary Plot (Global Importance)")
        shap_vals = explainer.shap_values(X)
        shap.summary_plot(shap_vals, X, feature_names=feature_names, show=False)
        fig = plt.gcf()
        summary_img_path = "shap_summary.png"
        fig.savefig(summary_img_path, bbox_inches='tight')
        st.pyplot(fig)

        with open(summary_img_path, "rb") as f:
            st.download_button(
                label="📥 Download SHAP Summary Plot (PNG)",
                data=f,
                file_name="shap_summary.png",
                mime="image/png"
            )
