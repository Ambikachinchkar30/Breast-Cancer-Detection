import streamlit as st
import numpy as np
import pickle

# Custom CSS styles for fonts, background, and button
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .main-title {
        font-size: 48px;
        color: #FF4B4B;
        text-align: center;
        font-weight: bold;
        font-family: 'Segoe UI', sans-serif;
    }
    .subtitle {
        font-size: 20px;
        text-align: center;
        color: #333333;
        margin-bottom: 30px;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton > button {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 12px;
        height: 50px;
        width: 200px;
        font-size: 18px;
        transition: all 0.3s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #ff1a1a;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# Load model
with open('svm_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Title and subtitle
st.markdown('<div class="main-title">Breast Cancer Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">🔍 Enter patient data below to predict diagnosis using SVM</div>', unsafe_allow_html=True)

# Define input features
feature_names = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

# Input fields in 3 columns
inputs = []
cols = st.columns(3)

for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        val = st.number_input(f"{feature.replace('_', ' ').title()}", format="%.6f")
        inputs.append(val)

# Predict button
st.markdown("###")
center_col = st.columns(3)[1]
with center_col:
    if st.button("🔮 Predict Diagnosis"):
        input_array = np.array([inputs])
        prediction = model.predict(input_array)[0]

        if prediction == 1:
            st.error("💥 The model predicts: **Malignant (Cancerous)**")
        else:
            st.success("✅ The model predicts: **Benign (Non-cancerous)**")
