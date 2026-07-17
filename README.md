# 🎗️ Breast Cancer Detection – Streamlit App

This project is part of my **100 Days of Buildables Fellowship**, where I build practical, end-to-end machine learning projects with real deployment workflows.
This app predicts whether a tumor is **Malignant (M)** or **Benign (B)** using the **Breast Cancer Wisconsin Diagnostic Dataset**.

---

## 🚀 Current Project – Breast Cancer Prediction App

**🌐 Live Demo**
👉 View on Streamlit Cloud: *([Breast Cancer Detection](https://100daysofbuildables-d7vsvtp5wwauwvruu2qk7i.streamlit.app/))*

---

## 📂 Repository Structure

```
Breast_Cancer_Prediction_Model/
│
├── app_bc.py                   # Streamlit dashboard for real-time predictions  
├── Project_2.ipynb             # Detailed notebook with preprocessing, EDA, tuning, and evaluation  
├── _model.pkl                  # Trained and optimized XGBoost model  
├── data.csv                    # Breast Cancer dataset  
├── requirements.txt            # Project dependencies  
└── README.md                   # Project documentation  
```

---

## 🛠️ Getting Started

Follow these steps to run the project locally:

### 1. **Clone the repository**

```bash
git clone https://github.com/AqsaNajeeb/100DaysOfBuildables/Breast_Cancer_Prediction_Model.git
cd Breast_Cancer_Prediction_Model
```

### 2. **Install dependencies**

It’s recommended to use a virtual environment:

```bash
pip install -r requirements.txt
```

### 3. **Run the app**

```bash
streamlit run app_bc.py
```

The app will open automatically in your browser at [http://localhost:8501](http://localhost:8501)

---

## ✅ Requirements

* Python 3.9+
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Plotly
* Joblib

All dependencies are listed in `requirements.txt`.

---

## 🌟 About This Project

This project is part of my **100 Days of Buildables** series to:

* Build a full ML pipeline — from EDA to deployment
* Apply **XGBoost** for highly accurate classification
* Perform **K-Fold Cross-Validation** and **Hyperparameter Tuning**
* Design a **Streamlit Dashboard** with interactive visualizations and prediction results
* Create a **real-time, deployable web app** for cancer diagnosis support

---

## 📊 Features

✅ Clean and interactive dashboard built in Streamlit

✅ Real-time tumor classification (Malignant / Benign)

✅ In-depth exploratory data analysis (EDA) with advanced visualizations

✅ Outlier detection and handling for robust predictions

✅ Optimized XGBoost model with tuned hyperparameters

✅ Downloadable trained model (`_model.pkl`) for reuse

---

## 📚 My Reflection

### What Improved After Tuning

* The **XGBoost** model achieved **higher accuracy and F1-score** after hyperparameter tuning.
* K-Fold Cross-Validation ensured **consistent performance** and reduced overfitting.
* Fine-tuning parameters such as `learning_rate`, `max_depth`, and `n_estimators` improved **model generalization**.

### Challenges Faced

* Managing **outliers** and ensuring proper **scaling** across numeric features required multiple iterations.
* The tuning process was **time-consuming**, especially with multiple parameter combinations.
* Selecting the **most influential features** for model interpretability.

### Key Insights from the Tuning

* XGBoost outperformed Random Forest in **precision and recall balance**.
* Cross-validation gave a more **trustworthy performance metric** compared to a single split.
* Preprocessing quality (handling missing values, normalization) had a major impact on the final outcome.
* Hyperparameter tuning enhanced both **accuracy** and **robustness**, preparing the model for deployment.

---

## 🤝 Connect With Me

If you’d like to collaborate, discuss ideas, or share feedback, feel free to reach out:

* **GitHub:** [Aqsa Najeeb](https://github.com/AqsaNajeeb)
* **LinkedIn:** [Aqsa Najeeb](https://www.linkedin.com/in/aqsa-najeeb/)

---

✨ If you like this project, don’t forget to **star ⭐ the repo!**
