# 🔮 Customer Churn Prediction System

An end-to-end **Machine Learning powered web application** that predicts customer churn probability and provides actionable insights to improve customer retention.  
Built using **TensorFlow, Scikit-learn, and Streamlit**, with a modern, responsive UI.

---

## 📌 Project Overview

Customer churn is one of the biggest challenges for businesses, especially in banking and subscription-based services.  
This project helps organizations **identify customers at risk of leaving** and take proactive retention actions.

The application allows users to input customer details such as demographics, account information, and activity status, then predicts:

- **Churn Probability**
- **Risk Level (High / Low)**
- **Key factors contributing to churn**

The goal is not just prediction, but **decision support**.

---

## 🚀 Key Features

- 📊 **Churn Probability Prediction** using a trained Neural Network  
- ⚠️ **Risk Classification** (High Risk / Low Risk customers)
- 📈 **Feature Impact Visualization**
  - Interactive Bar Chart 
- 🧠 **Heuristic Explainability** for churn drivers
- 🎨 **Modern UI**
  - Centered layout
  - Dark & Light mode support
  - Custom CSS styling
- ⚡ **Fast & Lightweight** Streamlit app
- ♻️ **Reusable ML Pipeline** (Scaler, Encoders, Model)

---

## 🛠️ Tech Stack

### 🔹 Machine Learning
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pandas

### 🔹 Web Application
- Streamlit
- Plotly (Interactive visualizations)
- Matplotlib

### 🔹 Model Artifacts
- Trained Neural Network (`.h5`)
- Label Encoder
- One-Hot Encoder
- Standard Scaler

---

## 📂 Project Structure

```text
├── model.h5
├── scaler.pkl
├── label_encoder_gender.pkl
├── onehot_encoder_geo.pkl
├── app.py
├── requirements.txt
└── README.md
