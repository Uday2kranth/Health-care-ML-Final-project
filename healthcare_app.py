"""
Healthcare ML Analytics Dashboard

A comprehensive machine learning application for healthcare data analysis
featuring breast cancer classification and diabetes risk prediction models.

Author: Healthcare Analytics Team
Version: 1.0.0
Last Updated: 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBClassifier, XGBRegressor
import warnings
import os
from PIL import Image

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Healthcare ML Analytics Dashboard",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
<style>
    /* Global text styling for better visibility */
    .main {
        color: white;
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 25%, #fd79a8 50%, #e17055 75%, #fdcb6e 100%);
        min-height: 100vh;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #6c5ce7 0%, #a29bfe 50%, #fd79a8 100%);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        font-size: 1.8rem;
        color: white;
        font-weight: bold;
        margin-bottom: 1rem;
        border-bottom: 3px solid #6c5ce7;
        padding-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 50%, #3d566e 100%);
        padding: 1.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
        border: 2px solid #ff7675;
        transition: transform 0.3s ease;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.2);
        border-color: #fd79a8;
    }
    
    .metric-card h3 {
        color: #ecf0f1 !important;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    .metric-card h2 {
        color: white !important;
        font-size: 2.2rem;
        margin-top: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .info-box {
        background: linear-gradient(135deg, #e8f5e8 0%, #f1f8e9 50%, #fff3e0 100%);
        padding: 1.2rem;
        border-radius: 15px;
        color: #2d3436;
        margin: 0.5rem 0;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        font-weight: 500;
        border: 2px solid #00b894;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .info-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
    }
    
    .prediction-result {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        margin: 1.5rem 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        border: 3px solid;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .prediction-result:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
    }
    
    .positive-result {
        background: linear-gradient(135deg, #ffe0e0 0%, #ffb3b3 100%);
        color: #c0392b;
        border-color: #e74c3c;
    }
    
    .negative-result {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        color: #27ae60;
        border-color: #2ecc71;
    }
    
    .moderate-result {
        background: linear-gradient(135deg, #FFC3C3 0%, #FFB3B3 100%);
        color: #c0392b;
        border-color: #e74c3c;
    }
    
    .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 50%, #2c3e50 100%);
        padding: 1.5rem;
        border-radius: 20px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        font-weight: 500;
        border: 2px solid #ff7675;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .sidebar-content:hover {
        transform: translateX(5px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
    }
    
    /* Enhanced button styling */
    .stButton > button {
        background: linear-gradient(45deg, #ff7675 0%, #fd79a8 100%);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 0.8rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #fd79a8 0%, #ff7675 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #ff7675 0%, #fd79a8 100%);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: linear-gradient(90deg, #2c3e50 0%, #34495e 100%);
        color: white;
        border: 2px solid #ff7675;
        border-radius: 10px;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        transform: translateY(-2px);
        border-color: #fd79a8;
    }
    
    .stSelectbox > div > div > div {
        color: white !important;
        font-weight: bold;
    }
    
    .stSelectbox > div > div > div[data-baseweb="select"] {
        background: linear-gradient(90deg, #2c3e50 0%, #34495e 100%);
        color: white;
    }
    
    .stSelectbox > div > div > div[data-baseweb="select"] > div {
        color: white !important;
    }
    
    /* Main container background */
    .main .block-container {
        background: linear-gradient(135deg, #ffffff 0%, #fff8f0 50%, #ffe4e1 100%);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Text elements for better contrast */
    .stMarkdown {
        color: white;
    }
    
    /* Enhance data tables */
    .dataframe {
        background: linear-gradient(135deg, #ffffff 0%, #fff8f0 100%);
        border-radius: 15px;
        border: 2px solid #ff7675;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .dataframe:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(90deg, #2c3e50 0%, #34495e 100%);
        border-radius: 15px;
        border: 2px solid #ff7675;
        transition: transform 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab-list"]:hover {
        transform: translateY(-2px);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.1);
        color: #fd79a8 !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: white !important;
        background: linear-gradient(90deg, #34495e 0%, #2c3e50 100%);
    }
    
    /* Enhanced sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    /* Plot container styling */
    .js-plotly-plot {
        border-radius: 20px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        border: 2px solid #ff7675;
        background: linear-gradient(135deg, #ffffff 0%, #fff8f0 100%);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .js-plotly-plot:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    
    /* Enhanced typography */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    p, div, span {
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    /* Custom styled sections */
    .section-header {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        border: 2px solid #ff7675;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    }
    
    .section-header:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
        border-color: #fd79a8;
    }
    
    /* Slider hover effects */
    .stSlider {
        transition: transform 0.3s ease;
    }
    
    .stSlider:hover {
        transform: translateY(-2px);
    }
    
    /* Image hover effects */
    .stImage > img {
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-radius: 15px;
    }
    
    .stImage:hover > img {
        transform: scale(1.02);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
</style>
""",
    unsafe_allow_html=True,
)


def load_image_from_resources(image_name):
    """
    Load image from resources folder if it exists

    Args:
        image_name (str): Name of the image file to load

    Returns:
        PIL.Image or None: Loaded image or None if not found
    """
    try:
        resources_path = os.path.join(os.getcwd(), "resources")
        image_path = os.path.join(resources_path, image_name)
        if os.path.exists(image_path):
            return Image.open(image_path)
        return None
    except Exception as e:
        st.warning(f"Could not load image: {image_name}")
        return None


@st.cache_data
def load_and_prepare_data():
    """
    Load and prepare both datasets for analysis

    Returns:
        tuple: (breast_cancer_df, diabetes_df, breast_cancer_raw, diabetes_raw)
    """
    # Load Breast Cancer Dataset
    breast_cancer = load_breast_cancer()
    bc_df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    bc_df["target"] = breast_cancer.target

    # Load Diabetes Dataset
    diabetes = load_diabetes()
    diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    diabetes_df["target"] = diabetes.target

    return bc_df, diabetes_df, breast_cancer, diabetes


@st.cache_resource
def train_breast_cancer_model(X, y):
    """Train breast cancer classification model"""
    # Select important features
    important_features = [
        "worst radius",
        "worst perimeter",
        "mean concave points",
        "worst concave points",
        "worst area",
    ]
    X_selected = X[important_features]

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )
    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Get predictions and metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, X_selected, accuracy, important_features


@st.cache_resource
def train_diabetes_model(X, y):
    """Train diabetes regression model"""
    model = XGBRegressor(random_state=42)
    model.fit(X, y)

    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    return model, mse, r2


def main():
    # Main header
    st.markdown(
        '<h1 class="main-header">Healthcare ML Analytics Dashboard</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="text-align: center; font-size: 1.3rem; color: white; margin-bottom: 2rem; font-weight: 500; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">Advanced Machine Learning for Healthcare Predictions & Analysis</p>',
        unsafe_allow_html=True,
    )

    # Load and display header image if available
    header_image = load_image_from_resources("header_image.jpg")
    if header_image:
        st.image(
            header_image,
            use_container_width=True,
            caption="Healthcare Analytics Dashboard",
        )

    # Load data
    bc_df, diabetes_df, breast_cancer_raw, diabetes_raw = load_and_prepare_data()

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown("## Model Selection")
        st.markdown("Choose a healthcare prediction model:")

        model_choice = st.selectbox(
            "Select Model:",
            [
                "Breast Cancer Classification",
                "Diabetes Risk Analysis",
                "Data Exploration",
            ],
            key="model_selector",
        )

        st.markdown("---")
        st.markdown("### Project Info")
        st.markdown("**Models:** XGBoost ML Algorithms")
        st.markdown("**Features:** Interactive Predictions")
        st.markdown("**Data:** Medical Datasets")
        st.markdown("</div>", unsafe_allow_html=True)

    # Main content based on selection
    if model_choice == "Breast Cancer Classification":
        show_breast_cancer_model(bc_df, breast_cancer_raw)
    elif model_choice == "Diabetes Risk Analysis":
        show_diabetes_model(diabetes_df, diabetes_raw)
    else:
        show_data_exploration(bc_df, diabetes_df)


def show_breast_cancer_model(bc_df, breast_cancer_raw):
    st.markdown(
        '<h2 class="sub-header">Breast Cancer Classification Model</h2>',
        unsafe_allow_html=True,
    )

    # Load and display model image if available
    model_image = load_image_from_resources("breast_cancer_model.jpg")
    if model_image:
        st.image(
            model_image,
            use_container_width=True,
            caption="Breast Cancer Classification Analysis",
        )

    # Train model
    X_bc = bc_df.drop("target", axis=1)
    y_bc = bc_df["target"]
    model, X_selected, accuracy, important_features = train_breast_cancer_model(
        X_bc, y_bc
    )

    # Model metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>Model Accuracy</h3>
            <h2>{accuracy:.1%}</h2>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>Features Used</h3>
            <h2>{len(important_features)}</h2>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>Total Samples</h3>
            <h2>{len(bc_df)}</h2>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Feature importance visualization
    st.markdown(
        '<div class="section-header">Feature Importance Analysis</div>',
        unsafe_allow_html=True,
    )
    feature_importance = pd.DataFrame(
        {"Feature": important_features, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=True)

    fig = px.bar(
        feature_importance,
        x="Importance",
        y="Feature",
        title="Feature Importance in Breast Cancer Prediction",
        color="Importance",
        color_continuous_scale="viridis",
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Interactive prediction
    st.markdown(
        '<div class="section-header">Make a Prediction</div>', unsafe_allow_html=True
    )
    st.markdown("Enter the patient's medical measurements:")

    col1, col2 = st.columns(2)

    with col1:
        worst_radius = st.slider("Worst Radius", 0.0, 50.0, 20.0, 0.1)
        mean_concave_points = st.slider("Mean Concave Points", 0.0, 0.3, 0.1, 0.01)
        worst_area = st.slider("Worst Area", 0.0, 4000.0, 1000.0, 10.0)

    with col2:
        worst_perimeter = st.slider("Worst Perimeter", 0.0, 300.0, 100.0, 1.0)
        worst_concave_points = st.slider("Worst Concave Points", 0.0, 0.5, 0.2, 0.01)

    if st.button("Predict Cancer Risk", type="primary", use_container_width=True):
        # Make prediction
        input_data = pd.DataFrame(
            {
                "worst radius": [worst_radius],
                "worst perimeter": [worst_perimeter],
                "mean concave points": [mean_concave_points],
                "worst concave points": [worst_concave_points],
                "worst area": [worst_area],
            }
        )

        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]

        # Display result
        if prediction == 1:
            st.markdown(
                f"""
            <div class="prediction-result negative-result">
                Prediction: BENIGN<br>
                Confidence: {prediction_proba[1]:.1%}<br>
                <small>Low risk of malignancy</small>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
            <div class="prediction-result positive-result">
                Prediction: MALIGNANT<br>
                Confidence: {prediction_proba[0]:.1%}<br>
                <small>Requires immediate medical attention</small>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Probability chart
        prob_df = pd.DataFrame(
            {"Class": ["Malignant", "Benign"], "Probability": prediction_proba}
        )

        fig = px.bar(
            prob_df,
            x="Class",
            y="Probability",
            title="Prediction Confidence",
            color="Probability",
            color_continuous_scale="RdYlBu",
        )
        st.plotly_chart(fig, use_container_width=True)


def show_diabetes_model(diabetes_df, diabetes_raw):
    st.markdown(
        '<h2 class="sub-header">Diabetes Risk Analysis Model</h2>',
        unsafe_allow_html=True,
    )

    # Load and display model image if available
    model_image = load_image_from_resources("diabetes_model.jpg")
    if model_image:
        st.image(
            model_image, use_container_width=True, caption="Diabetes Risk Analysis"
        )

    # Train model
    X_diabetes = diabetes_df.drop("target", axis=1)
    y_diabetes = diabetes_df["target"]
    model, mse, r2 = train_diabetes_model(X_diabetes, y_diabetes)

    # Model metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>R² Score</h3>
            <h2>{r2:.3f}</h2>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>MSE</h3>
            <h2>{mse:.1f}</h2>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>Features</h3>
            <h2>{X_diabetes.shape[1]}</h2>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Model performance visualization
    st.markdown(
        '<div class="section-header">Model Performance</div>', unsafe_allow_html=True
    )
    y_pred = model.predict(X_diabetes)

    fig = px.scatter(
        x=y_diabetes,
        y=y_pred,
        title="Diabetes Prediction: Actual vs Predicted",
        labels={"x": "Actual Values", "y": "Predicted Values"},
        color=y_pred,
        color_continuous_scale="viridis",
    )

    # Add perfect prediction line
    min_val, max_val = (
        min(y_diabetes.min(), y_pred.min()),
        max(y_diabetes.max(), y_pred.max()),
    )
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Perfect Prediction",
            line=dict(color="red", dash="dash"),
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Interactive prediction
    st.markdown(
        '<div class="section-header">Diabetes Risk Assessment</div>',
        unsafe_allow_html=True,
    )
    st.markdown("Enter patient health metrics:")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age Factor", -0.1, 0.1, 0.0, 0.01)
        sex = st.slider("Sex Factor", -0.1, 0.1, 0.0, 0.01)
        bmi = st.slider("BMI Factor", -0.1, 0.2, 0.0, 0.01)
        bp = st.slider("Blood Pressure", -0.1, 0.2, 0.0, 0.01)
        s1 = st.slider("Serum 1", -0.2, 0.2, 0.0, 0.01)

    with col2:
        s2 = st.slider("Serum 2", -0.2, 0.2, 0.0, 0.01)
        s3 = st.slider("Serum 3", -0.2, 0.2, 0.0, 0.01)
        s4 = st.slider("Serum 4", -0.2, 0.2, 0.0, 0.01)
        s5 = st.slider("Serum 5", -0.2, 0.2, 0.0, 0.01)
        s6 = st.slider("Serum 6", -0.2, 0.2, 0.0, 0.01)

    if st.button("Assess Diabetes Risk", type="primary", use_container_width=True):
        # Make prediction
        input_data = pd.DataFrame(
            {
                "age": [age],
                "sex": [sex],
                "bmi": [bmi],
                "bp": [bp],
                "s1": [s1],
                "s2": [s2],
                "s3": [s3],
                "s4": [s4],
                "s5": [s5],
                "s6": [s6],
            }
        )

        prediction = model.predict(input_data)[0]

        # Interpret prediction
        if prediction < 100:
            risk_level = "Low"
            color_class = "negative-result"
            icon = ""
        elif prediction < 200:
            risk_level = "Moderate"
            color_class = "moderate-result"
            icon = ""
        else:
            risk_level = "High"
            color_class = "positive-result"
            icon = ""

        st.markdown(
            f"""
        <div class="prediction-result {color_class}">
            {icon} Diabetes Risk Score: {prediction:.1f}<br>
            Risk Level: {risk_level}<br>
            <small>Score range: 25-346 (higher = more risk)</small>
        </div>
        """,
            unsafe_allow_html=True,
        )


def show_data_exploration(bc_df, diabetes_df):
    st.markdown(
        '<h2 class="sub-header">Healthcare Data Exploration</h2>',
        unsafe_allow_html=True,
    )

    # Load and display data exploration image if available
    data_image = load_image_from_resources("data_exploration.jpg")
    if data_image:
        st.image(
            data_image, use_container_width=True, caption="Healthcare Data Analysis"
        )

    # Dataset overview
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Breast Cancer Dataset")
        st.markdown(
            f"""
        <div class="info-box">
            <strong>Samples:</strong> {len(bc_df)}<br>
            <strong>Features:</strong> {bc_df.shape[1]-1}<br>
            <strong>Target Classes:</strong> Malignant/Benign<br>
            <strong>Class Distribution:</strong><br>
            • Benign: {sum(bc_df['target'] == 1)} ({sum(bc_df['target'] == 1)/len(bc_df)*100:.1f}%)<br>
            • Malignant: {sum(bc_df['target'] == 0)} ({sum(bc_df['target'] == 0)/len(bc_df)*100:.1f}%)
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Breast cancer distribution
        fig1 = px.pie(
            values=[sum(bc_df["target"] == 0), sum(bc_df["target"] == 1)],
            names=["Malignant", "Benign"],
            title="Breast Cancer Class Distribution",
            color_discrete_sequence=["#ff6b6b", "#4ecdc4"],
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("### Diabetes Dataset")
        st.markdown(
            f"""
        <div class="info-box">
            <strong>Samples:</strong> {len(diabetes_df)}<br>
            <strong>Features:</strong> {diabetes_df.shape[1]-1}<br>
            <strong>Target:</strong> Continuous (25-346)<br>
            <strong>Mean Target:</strong> {diabetes_df['target'].mean():.1f}<br>
            <strong>Std Target:</strong> {diabetes_df['target'].std():.1f}
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Diabetes distribution
        fig2 = px.histogram(
            diabetes_df,
            x="target",
            title="Diabetes Target Distribution",
            nbins=30,
            color_discrete_sequence=["#667eea"],
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Feature correlations
    st.markdown(
        '<div class="section-header">Feature Correlations</div>', unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs(["Breast Cancer Correlations", "Diabetes Correlations"])

    with tab1:
        # Select top features for breast cancer
        important_bc_features = [
            "worst radius",
            "worst perimeter",
            "mean concave points",
            "worst concave points",
            "worst area",
            "target",
        ]
        bc_corr = bc_df[important_bc_features].corr()

        fig = px.imshow(
            bc_corr,
            text_auto=True,
            aspect="auto",
            title="Breast Cancer Feature Correlations",
            color_continuous_scale="RdBu",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        diabetes_corr = diabetes_df.corr()

        fig = px.imshow(
            diabetes_corr,
            text_auto=True,
            aspect="auto",
            title="Diabetes Feature Correlations",
            color_continuous_scale="RdBu",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Summary statistics
    st.markdown(
        '<div class="section-header">Summary Statistics</div>', unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Breast Cancer Dataset Summary:**")
        st.dataframe(bc_df.describe().round(2))

    with col2:
        st.markdown("**Diabetes Dataset Summary:**")
        st.dataframe(diabetes_df.describe().round(2))


# Footer
def show_footer():
    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center; color: white; padding: 2rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);'>
        <h4 style='color: white; margin-bottom: 1rem;'>Healthcare ML Analytics Dashboard</h4>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
    show_footer()
