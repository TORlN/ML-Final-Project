# Heart Disease Prediction - ML Final Project

A machine learning project to predict the presence of heart disease in patients based on medical and demographic features using binary classification algorithms.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)

## 🎯 Project Overview

This project aims to build and evaluate machine learning models that can accurately predict whether a patient has heart disease based on various medical measurements and patient demographics. The goal is to create a reliable classification system that could assist healthcare professionals in early detection and diagnosis.

### Problem Type
- **Binary Classification**: Predict presence (1) or absence (0) of heart disease
- **Target Variable**: Heart disease presence indicator
- **Success Criteria**: High accuracy with minimal false negatives (missing actual heart disease cases)

## 📊 Dataset

The project uses the Heart Disease dataset from Kaggle, which contains medical records with the following characteristics:

- **Source**: [Heart Disease Dataset on Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Size**: Multiple patient records with medical measurements
- **Features**: Various medical measurements and patient demographics including:
  - Age, sex, chest pain type (cp)
  - Resting blood pressure, cholesterol levels
  - Fasting blood sugar, ECG results
  - Maximum heart rate, exercise-induced angina
  - ST depression, slope, and other cardiac indicators

### Target Variable
- **0**: No heart disease
- **1**: Heart disease present

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup
1. Clone or download this repository:
   ```bash
   git clone https://github.com/TORlN/ML-Final-Project.git
   cd ML-Final-Project
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the dataset is in the correct location:
   ```
   data/heart.csv
   ```

## 💻 Usage

### Running the Analysis
1. Open the Jupyter notebook:
   ```bash
   jupyter notebook Final_Project.ipynb
   ```

2. Run all cells sequentially to:
   - Load and inspect the data
   - Perform exploratory data analysis
   - Train machine learning models
   - Evaluate model performance

### Key Sections
- **Data Loading & Inspection**: Initial data exploration and quality checks
- **EDA & Visualization**: Statistical analysis and data visualization
- **Data Preprocessing**: Feature engineering and data preparation
- **Model Training**: Implementation of Logistic Regression and Random Forest
- **Evaluation**: Performance metrics and model comparison

## 📁 Project Structure

```
ML Final Project/
├── Final_Project.ipynb          # Main analysis notebook
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── 📅 Project Timeline & Milestones.md  # Project planning document
└── data/
    └── heart.csv               # Heart disease dataset
```

## 🔬 Methodology

### 1. Problem Definition
- Defined binary classification objective
- Established evaluation metrics (Accuracy, Precision, Recall, F1-Score, AUC-ROC)

### 2. Data Analysis
- **Data Inspection**: Checked data types, missing values, and duplicates
- **Statistical Summary**: Generated descriptive statistics
- **Data Quality**: Verified data integrity and handled intentional duplicates

### 3. Exploratory Data Analysis
- **Distribution Analysis**: Examined continuous and categorical variables
- **Correlation Analysis**: Identified feature relationships
- **Visualization**: Created comprehensive plots for data understanding

### 4. Data Preprocessing
- **Feature Encoding**: Handled categorical variables
- **Data Splitting**: 80/20 train-test split
- **Scaling**: Applied StandardScaler for numerical features

### 5. Model Implementation
- **Logistic Regression**: Baseline linear model
- **Random Forest**: Ensemble tree-based model

### 6. Evaluation
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score
- **Visualization**: Confusion matrices and ROC curves
- **Feature Importance**: Analysis of most predictive features

## 📈 Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~82% | Good | Good | Good | Good |
| Random Forest | 100% | 100% | 100% | 100% | 1.00 |

### Key Observations
- **Random Forest achieved perfect performance** on the test set
- **No data leakage detected** - verified through multiple checks
- **Strong categorical predictors** in the dataset enable excellent tree-based model performance

## 🔍 Key Findings

1. **Feature Importance**: Categorical variables (chest pain type, thalassemia, coronary arteries) are highly predictive
2. **Model Comparison**: Tree-based models significantly outperform linear models on this dataset
3. **Data Quality**: The Kaggle dataset combines multiple heart disease datasets with intentional resampling
4. **Perfect Performance**: The 100% Random Forest accuracy appears genuine due to the highly predictive categorical features

## 🛠 Technologies Used

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning algorithms and evaluation
- **Jupyter Notebook**: Interactive development environment
