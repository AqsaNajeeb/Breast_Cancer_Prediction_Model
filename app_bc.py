# -*- coding: utf-8 -*-
"""app_bc.py — Breast Cancer Diagnosis Dashboard"""

# ============================================
# 🧬 Breast Cancer Diagnosis Prediction Dashboard
# Using XGBoost + Streamlit + Built-in Dataset
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
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
        .metric-card {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            text-align: center;
        }

        /* Make tabs larger, clearer, and visually appealing */
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
    </style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================
st.markdown('<h1 class="main-title">🧬 Breast Cancer Diagnosis Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Built with Streamlit | Powered by XGBoost | Embedded Dataset</p>', unsafe_allow_html=True)
st.markdown("---")

# ============================================
# LOAD DATASET
# ============================================
DATA_PATH = "data.csv"

if not os.path.exists(DATA_PATH):
    st.error("⚠️ Dataset file 'data.csv' not found. Please make sure it exists in the project folder.")
    st.stop()

data = pd.read_csv(DATA_PATH)
data = data.drop(columns=[c for c in ["Unnamed: 32", "id"] if c in data.columns], errors='ignore')

# Encode target if categorical
if data["diagnosis"].dtype == "object":
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

# ============================================
# LOAD TRAINED MODEL (JOBLIB ONLY)
# ============================================
MODEL_PATH = "_model.pkl"

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
    st.error("⚠️ Model file '_model.pkl' not found or empty. Please train and save it first using joblib.dump().")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Failed to load model file:\n\n**{e}**")
    st.stop()

# ============================================
# MODEL EVALUATION
# ============================================
try:
    y_pred = model.predict(X)
except Exception as e:
    st.error(f"⚠️ Error during prediction:\n\n{e}")
    st.stop()

accuracy = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred)
cm = confusion_matrix(y, y_pred)

# ============================================
# DASHBOARD SUMMARY CARDS
# ============================================
st.markdown("""
    <style>
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

st.markdown("### 📊 Model Overview Summary")

col1, col2, col3, col4 = st.columns(4)
col1.markdown(f"<div class='metric-card'><h3>Dataset Size</h3><h2>{data.shape[0]}</h2><p>Samples</p></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-card'><h3>Features</h3><h2>{data.shape[1]-1}</h2><p>Input Variables</p></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-card'><h3>Accuracy</h3><h2>{accuracy*100:.2f}%</h2><p>Model Performance</p></div>", unsafe_allow_html=True)
col4.markdown(f"<div class='metric-card'><h3>F1 Score</h3><h2>{f1:.2f}</h2><p>Balanced Measure</p></div>", unsafe_allow_html=True)

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

    # --- Top Section: Text + Confusion Matrix ---
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("""
        This dashboard predicts whether a breast tumor is **Malignant** or **Benign**
        using a fine-tuned **XGBoost Classifier** trained on diagnostic cell data.
        
        The overview below highlights the model's performance through its predictions,
        confidence levels, and output distribution.
        """)
    with col2:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
        disp.plot(ax=ax, cmap="Purples", colorbar=False)
        plt.title("Confusion Matrix")
        st.pyplot(fig)

    # --- Bottom Section: Two Tiny Side-by-Side Graphs ---
    st.markdown("### 📈 Model Confidence & Output Distribution")

    col3, col4 = st.columns(2)

    # --- Confidence Distribution (Left) ---
    with col3:
        y_proba = model.predict_proba(X)[:, 1]
        fig, ax = plt.subplots(figsize=(3.8, 3.8))
        sns.histplot(y_proba, bins=15, kde=True, color="#9370db", ax=ax)
        plt.xlabel("Confidence (Malignant)", fontsize=8)
        plt.ylabel("Count", fontsize=8)
        plt.title("Confidence Levels", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    # --- Prediction Outcome Pie Chart (Right) ---
    with col4:
        pred_counts = pd.Series(y_pred).value_counts()
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

    st.markdown("<p style='text-align:center; color:gray; font-size:14px;'>Developed by <b>Aqsa Najeeb</b> | Powered by Streamlit + XGBoost</p>", unsafe_allow_html=True)


# ============================================
# TAB 2: DATA INSIGHTS (Feature Importance moved here)
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
        sns.heatmap(data.corr(), cmap="Purples", center=0)
        plt.title("Feature Correlation Matrix")
        st.pyplot(fig)

    st.markdown("### 🧠 Top Correlated Features")
    corr_target = data.corr()["diagnosis"].sort_values(ascending=False)[1:11]
    st.bar_chart(corr_target)

    # === FEATURE IMPORTANCE SECTION (Moved here only) ===
    st.markdown("### 🔑 Feature Importance (XGBoost)")

    xgb_model = None
    if isinstance(model, XGBClassifier):
        xgb_model = model
    elif hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if isinstance(step, XGBClassifier):
                xgb_model = step
                break

    if xgb_model is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        xgb.plot_importance(xgb_model, ax=ax, importance_type="weight", color="purple")
        plt.title("Top Important Features")
        st.pyplot(fig)
    else:
        st.warning("⚠️ Could not extract XGBClassifier for feature importance plot.")

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
        input_df = pd.DataFrame([input_data])
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
