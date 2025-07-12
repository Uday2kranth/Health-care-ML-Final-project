# Healthcare ML Analytics Dashboard

A comprehensive machine learning project for healthcare data analysis featuring breast cancer classification and diabetes risk prediction models with an interactive web dashboard.

## Live Demo

**Try the Live Application:**
- **Streamlit App**: [Healthcare ML Dashboard](https://udaykranthi-health-care-ml-final-projectgitday.streamlit.app/)
- **Repository**: [GitHub Repository](https://github.com/Uday2kranth/Health-care-ML-Final-project)

## Project Overview

This repository contains a complete healthcare machine learning solution with two main components:

1. **Jupyter Notebook Analysis** - Detailed exploratory data analysis and model development
2. **Streamlit Web Application** - Interactive dashboard for real-time predictions

## Repository Structure

```
healthcare-ml-dashboard/
├── Healthcare_ML_Models_Annotated_and_Visualized.ipynb  # Main analysis notebook
├── Healthcare_ML_Models_Organized.ipynb                 # Clean organized code
├── healthcare_app.py                                    # Streamlit web application
├── requirements.txt                                     # Python dependencies
├── resources/                                           # Images and assets
│   ├── README.md                                       # Image guidelines
│   ├── header_image.jpg                                # Dashboard header image
│   ├── breast_cancer_model.jpg                         # Breast cancer section image
│   ├── diabetes_model.jpg                              # Diabetes section image
│   └── data_exploration.jpg                            # Data exploration image
├── .gitignore                                          # Git ignore file
└── README.md                                           # This file
```

## Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Local Installation

1. **Clone or Download the Repository**
   ```bash
   git clone https://github.com/Uday2kranth/Health-care-ML-Final-project.git
   cd Health-care-ML-Final-project
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App**
   ```bash
   streamlit run healthcare_app.py
   ```

4. **Open in Browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in the terminal

### Cloud Deployment Options

#### **Streamlit Cloud (Recommended)**
1. Fork this repository
2. Sign up at [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Deploy with one click
5. Live app: [Healthcare ML Dashboard](https://udaykranthi-health-care-ml-final-projectgitday.streamlit.app/)

#### **Other Deployment Options**
- **Railway**: Simple deployment with GitHub integration
- **Render**: Free tier available for small projects
- **Google Cloud Platform**: Professional deployment option
- **AWS**: Enterprise-level deployment solution

##  Healthcare_ML_Models_Annotated_and_Visualized Notebook

This comprehensive Jupyter notebook provides a step-by-step analysis of healthcare datasets:

### **Step 1: Data Loading and Exploration**
- **Breast Cancer Dataset**: 569 samples, 30 features
- **Diabetes Dataset**: 442 samples, 10 features
- Initial data inspection and basic statistics

### **Step 2: Exploratory Data Analysis (EDA)**
- **Statistical Summary**: Mean, median, standard deviation for all features
- **Class Distribution**: Visualization of target variable distributions
- **Feature Correlations**: Correlation heatmaps to identify relationships
- **Data Visualization**: 
  - Histograms for feature distributions
  - Box plots for outlier detection
  - Scatter plots for feature relationships

### **Step 3: Data Preprocessing**
- **Missing Value Analysis**: Check for null values
- **Feature Selection**: Identify most important features
- **Data Scaling**: Normalize features for better model performance
- **Train-Test Split**: Prepare data for model training

### **Step 4: Model Development**

#### **Breast Cancer Classification Model**
- **Algorithm**: XGBoost Classifier
- **Features Used**: 
  - Worst radius
  - Worst perimeter
  - Mean concave points
  - Worst concave points
  - Worst area
- **Performance Metrics**:
  - Accuracy: ~95%
  - Precision, Recall, F1-Score
  - Confusion Matrix

#### **Diabetes Regression Model**
- **Algorithm**: XGBoost Regressor
- **Features Used**: All 10 diabetes features
- **Performance Metrics**:
  - R² Score: ~0.5
  - Mean Squared Error
  - Mean Absolute Error

### **Step 5: Model Visualization**
- **Feature Importance Plots**: Bar charts showing feature contributions
- **Confusion Matrix**: Visual representation of classification results
- **Prediction vs Actual**: Scatter plots for regression analysis
- **ROC Curves**: Model performance evaluation

### **Step 6: Model Evaluation**
- **Cross-validation**: Ensure model robustness
- **Performance Analysis**: Detailed metrics interpretation
- **Model Comparison**: Baseline vs optimized models

## Healthcare App (Streamlit Dashboard)

The interactive web application provides a user-friendly interface for healthcare predictions:

### **How the App Works**

#### **1. Application Structure**
```python
# Main components
├── load_and_prepare_data()     # Data loading and caching
├── train_breast_cancer_model() # Model training with caching
├── train_diabetes_model()      # Model training with caching
├── load_image_from_resources() # Image loading functionality
└── main()                      # Main application logic
```

#### **2. Sidebar Navigation**
- **Model Selection**: Choose between three main sections
- **Project Information**: Overview of technologies used
- **Responsive Design**: Clean, professional interface

#### **3. Main Sections**

##### **Breast Cancer Classification**
- **Input Interface**: Interactive sliders for patient measurements
- **Real-time Prediction**: Instant classification results
- **Confidence Scores**: Probability of benign vs malignant
- **Visual Analytics**: Feature importance charts

##### **Diabetes Risk Analysis**
- **Health Metrics Input**: Patient health parameter sliders
- **Risk Assessment**: Continuous risk score calculation
- **Performance Visualization**: Actual vs predicted scatter plots
- **Risk Interpretation**: Low, moderate, high risk categories

##### **Data Exploration**
- **Dataset Overview**: Summary statistics and distributions
- **Interactive Charts**: Pie charts, histograms, correlation heatmaps
- **Comparative Analysis**: Side-by-side dataset comparison

#### **4. Technical Features**
- **Caching**: `@st.cache_data` and `@st.cache_resource` for performance
- **Responsive Layout**: Multi-column layouts and tabs
- **Error Handling**: Graceful error management
- **Image Support**: Dynamic image loading from resources folder

### **UI/UX Features**
- **Modern Design**: Gradient backgrounds and professional styling
- **Interactive Elements**: Sliders, buttons, and dropdowns
- **Visual Feedback**: Color-coded prediction results
- **Responsive Layout**: Works on different screen sizes

## Step-by-Step Usage Instructions

### **For Repository Users (Fork/Download)**

#### **1. Environment Setup**
```bash
# After cloning/downloading
git clone https://github.com/Uday2kranth/Health-care-ML-Final-project.git
cd Health-care-ML-Final-project
pip install -r requirements.txt
```

#### **2. Running the Jupyter Notebook**
```bash
# Start Jupyter
jupyter notebook

# Open in browser
# Navigate to Healthcare_ML_Models_Annotated_and_Visualized.ipynb
# Run cells sequentially from top to bottom
```

#### **3. Running the Streamlit App**
```bash
# Start the web application
streamlit run healthcare_app.py

# Access in browser
# Navigate to http://localhost:8501
```

#### **4. Deploying Your Own Version**

##### **Quick Deploy to Streamlit Cloud**
1. Fork this repository to your GitHub account
2. Sign up at [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Click "Deploy app"
5. Live app: [Healthcare ML Dashboard](https://udaykranthi-health-care-ml-final-projectgitday.streamlit.app/)

##### **Alternative Deployment Options**
- **Railway**: Simple deployment with GitHub integration
- **Render**: Free tier available for small projects  
- **Google Cloud Platform**: Professional deployment option
- **AWS**: Enterprise-level deployment solution

#### **4. Using the Dashboard**

1. **Select Model** from sidebar:
   - Breast Cancer Classification
   - Diabetes Risk Analysis  
   - Data Exploration

2. **Input Data** using interactive sliders:
   - Adjust values based on patient measurements
   - See real-time updates

3. **Get Predictions**:
   - Click prediction buttons
   - View confidence scores and risk levels
   - Analyze visualization charts

4. **Explore Data**:
   - Review dataset statistics
   - Examine feature correlations
   - Compare model performances

#### **5. Customization Options**

##### **Adding Images**
```bash
# Add images to resources folder
resources/
├── header_image.jpg          # Main dashboard header
├── breast_cancer_model.jpg   # Breast cancer section
├── diabetes_model.jpg        # Diabetes section
└── data_exploration.jpg      # Data exploration section
```

##### **Modifying Models**
- Edit `train_breast_cancer_model()` for different features
- Adjust `train_diabetes_model()` for different algorithms
- Update visualization functions for custom charts

## Technical Stack

### **Machine Learning**
- **Scikit-learn**: Data preprocessing and metrics
- **XGBoost**: Gradient boosting models
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations

### **Visualization**
- **Plotly**: Interactive charts
- **Matplotlib**: Static plots
- **Seaborn**: Statistical visualizations

### **Web Framework**
- **Streamlit**: Web application framework
- **PIL (Pillow)**: Image processing

### **Development Tools**
- **Jupyter**: Interactive development
- **Git**: Version control

## Model Performance

### **Breast Cancer Classification**
- **Accuracy**: ~95%
- **Precision**: High precision for both classes
- **Recall**: Balanced recall scores
- **Features**: 5 most important features selected

### **Diabetes Risk Prediction**
- **R² Score**: ~0.5
- **MSE**: Reasonable prediction error
- **Feature Importance**: All 10 features contribute
- **Range**: Risk scores from 25-346

## Troubleshooting

### **Common Issues**

1. **Import Errors**
   ```bash
   # Ensure all packages are installed
   pip install -r requirements.txt
   ```

2. **Port Already in Use**
   ```bash
   # Use different port
   streamlit run healthcare_app.py --server.port 8502
   ```

3. **Image Loading Issues**
   ```bash
   # Check resources folder exists
   # Verify image file names match exactly
   ```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Deployment Links

**Live Application:**
- **Primary**: [Healthcare ML Dashboard](https://udaykranthi-health-care-ml-final-projectgitday.streamlit.app/) - *Streamlit Cloud Deployment*
- **Repository**: [GitHub Repository](https://github.com/Uday2kranth/Health-care-ML-Final-project) - *Source Code*

**Deployment Status:**
- [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://udaykranthi-health-care-ml-final-projectgitday.streamlit.app/)

## License

This project is for educational purposes. Please ensure compliance with healthcare data regulations in production use.

## Acknowledgments

- **Scikit-learn** for providing healthcare datasets
- **Streamlit** for the web framework
- **XGBoost** for high-performance machine learning
- **Plotly** for interactive visualizations

---

**Important Note**: This application is for educational and demonstration purposes only. Always consult qualified healthcare professionals for medical advice and diagnosis.

## Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code documentation
3. Open an issue on the repository
4. Consult the official documentation for used libraries

---

**Happy Healthcare Analytics!**
