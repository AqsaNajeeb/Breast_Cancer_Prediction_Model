# -*- coding: utf-8 -*-
"""app_bc.py — Breast Cancer Diagnosis Dashboard (fully self-contained)

This version does NOT read data.csv or _model.pkl from disk at runtime.
The dataset is loaded from scikit-learn's bundled Breast Cancer Wisconsin
(Diagnostic) dataset — the same underlying data as the Kaggle
"uciml/breast-cancer-wisconsin-data" CSV, just shipped inside the sklearn
package instead of as a separate file — and the Logistic Regression
pipeline is trained live, in-memory, the first time the app runs
(and then cached so it's instant afterwards).

Nothing external needs to exist next to this file for it to work.
"""

# ============================================
# 🧬 Breast Cancer Diagnosis Prediction Dashboard
# Using Logistic Regression (Pipeline) + Streamlit
# Dataset + Model are both embedded / trained in-app
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, ConfusionMatrixDisplay

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(page_title="Breast Cancer Dashboard", page_icon="🧬", layout="wide")

# ============================================
# STYLES
# ============================================
st.markdown("""
    <style>
        body {
            background: linear-gradient(120deg, #f5f8ff, #e9ecff);
        }
        .main-title {
            text-align: center;
            color: #4B0082;
            font-size: 40px;
            font-weight: 800;
            margin-bottom: 0;
        }
        .subheader {
            text-align: center;
            color: #555;
            font-size: 18px;
            margin-top: 5px;
        }
        .stTabs [data-baseweb="tab-list"] {
            justify-content: center;
            gap: 12px;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 18px;
            font-weight: 700;
            color: #4B0082 !important;
            padding: 12px 26px;
            border-radius: 8px !important;
            background-color: #f3f1ff;
            transition: all 0.3s ease-in-out;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e6dbff !important;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4B0082 !important;
            color: white !important;
        }
        .metric-card {
            background: linear-gradient(145deg, #e6e0ff, #ffffff);
            padding: 22px;
            border-radius: 14px;
            box-shadow: 0 6px 16px rgba(75, 0, 130, 0.15);
            text-align: center;
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 18px rgba(75, 0, 130, 0.25);
        }
        .metric-card h3 {
            color: #4B0082;
            font-size: 18px;
            margin-bottom: 6px;
        }
        .metric-card h2 {
            color: #2a004f;
            font-size: 28px;
            margin: 0;
        }
        .metric-card p {
            color: #555;
            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================
st.markdown('<h1 class="main-title">🧬 Breast Cancer Diagnosis Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Built with Streamlit | Powered by Logistic Regression | Dataset & Model Embedded</p>', unsafe_allow_html=True)
st.markdown("---")


# ============================================
# EMBEDDED DATASET (no data.csv needed)
# ============================================
@st.cache_data
def load_data() -> pd.DataFrame:
    """Loads the Wisconsin Breast Cancer diagnostic dataset that ships
    inside scikit-learn — identical measurements to the Kaggle CSV
    (radius/texture/perimeter/... mean, se, worst for each cell nucleus).
    """
    raw = load_breast_cancer()

    # sklearn's feature names look like "mean radius", "radius error",
    # "worst radius" — rename to match the familiar radius_mean,
    # radius_se, radius_worst convention from the original CSV.
    def rename(col: str) -> str:
        col = col.replace("concave points", "concave_points")
        if col.startswith("mean "):
            base = col.replace("mean ", "")
            suffix = "_mean"
        elif col.endswith(" error"):
            base = col.replace(" error", "")
            suffix = "_se"
        elif col.startswith("worst "):
            base = col.replace("worst ", "")
            suffix = "_worst"
        else:
            base = col
            suffix = ""
        return base.replace(" ", "_") + suffix

    df = pd.DataFrame(raw.data, columns=[rename(c) for c in raw.feature_names])

    # sklearn encodes target as 0=malignant, 1=benign — flip it so it
    # matches the original CSV convention (M=1, B=0).
    df["diagnosis"] = 1 - raw.target
    return df


data = load_data()
X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ============================================
# EMBEDDED MODEL (no _model.pkl needed)
# ============================================
@st.cache_resource
def train_model(X_train, y_train) -> Pipeline:
    """Trains the StandardScaler + LogisticRegression pipeline once and
    caches it in memory for the life of the app session/server.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


model = train_model(X_train, y_train)

# ============================================
# MODEL EVALUATION (on held-out test split)
# ============================================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Full-dataset predictions, used only for the Overview tab's confidence
# / breakdown charts so they reflect the whole embedded dataset.
y_pred_full = model.predict(X)
y_proba_full = model.predict_proba(X)[:, 1]

# ============================================
# DASHBOARD SUMMARY CARDS
# ============================================
st.markdown("### 📊 Model Overview Summary")

col1, col2, col3, col4 = st.columns(4)
col1.markdown(f"<div class='metric-card'><h3>Dataset Size</h3><h2>{data.shape[0]}</h2><p>Samples</p></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-card'><h3>Features</h3><h2>{data.shape[1]-1}</h2><p>Input Variables</p></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-card'><h3>Test Accuracy</h3><h2>{accuracy*100:.2f}%</h2><p>Model Performance</p></div>", unsafe_allow_html=True)
col4.markdown(f"<div class='metric-card'><h3>Test F1 Score</h3><h2>{f1:.2f}</h2><p>Balanced Measure</p></div>", unsafe_allow_html=True)

st.markdown("---")

# ============================================
# TABS
# ============================================
tab1, tab2, tab3 = st.tabs(["🏠 Overview", "📊 Data Insights", "🤖 Prediction Interface"])

# ============================================
# TAB 1: OVERVIEW
# ============================================
with tab1:
    st.header("🏠 Overview")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("""
        This dashboard predicts whether a breast tumor is **Malignant** or **Benign**
        using a **Logistic Regression** classifier (with feature scaling), trained on
        the Wisconsin Breast Cancer diagnostic dataset.

        Both the dataset and the trained model are embedded directly in this app —
        there's nothing to upload or configure. Metrics below are measured on a
        held-out 20% test split.
        """)
    with col2:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
        disp.plot(ax=ax, cmap="Purples", colorbar=False)
        plt.title("Confusion Matrix (Test Set)")
        st.pyplot(fig)

    st.markdown("### 📈 Model Confidence & Output Distribution")

    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots(figsize=(3.8, 3.8))
        sns.histplot(y_proba_full, bins=15, kde=True, color="#9370db", ax=ax)
        plt.xlabel("Confidence (Malignant)", fontsize=8)
        plt.ylabel("Count", fontsize=8)
        plt.title("Confidence Levels (Full Dataset)", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    with col4:
        pred_counts = pd.Series(y_pred_full).value_counts().reindex([0, 1]).fillna(0)
        fig, ax = plt.subplots(figsize=(3.8, 3.8))
        ax.pie(
            pred_counts,
            labels=["Benign", "Malignant"],
            autopct='%1.1f%%',
            colors=["#b19cd9", "#9370db"],
            startangle=90,
            textprops={'fontsize': 8}
        )
        plt.title("Prediction Breakdown", fontsize=9)
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("<p style='text-align:center; color:gray; font-size:14px;'>Developed by <b>Aqsa Najeeb</b> | Powered by Streamlit + Logistic Regression</p>", unsafe_allow_html=True)


# ============================================
# TAB 2: DATA INSIGHTS
# ============================================
with tab2:
    st.header("📊 Data Insights")
    st.markdown("### 🔍 Dataset Preview")
    st.dataframe(data.head(), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🎯 Target Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="diagnosis", data=data, palette="cool")
        plt.title("Benign vs Malignant Count")
        st.pyplot(fig)
    with c2:
        st.markdown("#### 🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(data.corr(numeric_only=True), cmap="Purples", center=0)
        plt.title("Feature Correlation Matrix")
        st.pyplot(fig)

    st.markdown("### 🧠 Top Correlated Features")
    corr_target = data.corr(numeric_only=True)["diagnosis"].sort_values(ascending=False)[1:11]
    st.bar_chart(corr_target)

    st.markdown("### 🔑 Feature Importance (Logistic Regression Coefficients)")

    logreg_step = model.named_steps.get("classifier")
    if logreg_step is not None:
        coefs = pd.Series(logreg_step.coef_[0], index=list(X.columns))
        top_coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).index)[:15]

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#9370db" if v > 0 else "#b19cd9" for v in top_coefs.values]
        ax.barh(top_coefs.index[::-1], top_coefs.values[::-1], color=colors[::-1])
        ax.set_xlabel("Coefficient (impact on Malignant probability)")
        ax.set_title("Top 15 Features by |Coefficient|")
        plt.tight_layout()
        st.pyplot(fig)

        st.caption("Positive bars push predictions toward **Malignant**; negative bars push toward **Benign**. "
                   "Coefficients are on the scaled-feature space (StandardScaler), so magnitudes are comparable across features.")
    else:
        st.warning("⚠️ Could not extract the LogisticRegression step for the coefficient plot.")

# ============================================
# TAB 3: PREDICTION INTERFACE
# ============================================
with tab3:
    st.header("🤖 Model Prediction Interface")

    cols = st.columns(3)
    input_data = {}
    for i, feature in enumerate(X.columns):
        with cols[i % 3]:
            input_data[feature] = st.number_input(
                f"{feature}",
                value=float(np.round(data[feature].mean(), 2))
            )

    if st.button("🔍 Predict"):
        input_df = pd.DataFrame([input_data])[list(X.columns)]

        if input_df.isna().any().any():
            st.error("⚠️ Please fill in all fields — one or more values are missing.")
            st.stop()

        try:
            pred = model.predict(input_df)[0]
        except Exception as e:
            st.error(f"⚠️ Prediction error:\n\n{e}")
            st.stop()

        label = "Malignant" if pred == 1 else "Benign"

        st.markdown("---")
        if label == "Malignant":
            st.error("🚨 **Result: Malignant Tumor Detected**")
        else:
            st.success("✅ **Result: Benign Tumor Detected**")

        st.markdown("### 📋 Input Summary")
        st.dataframe(input_df.T, use_container_width=True)
