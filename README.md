# Customer Churn Prediction Project

## Overview
This project aims to predict customer churn for a telecommunications company using advanced machine learning algorithms. The project applies various machine learning techniques to identify patterns in customer behavior that indicate a likelihood to discontinue services.

## Features
- Three different machine learning models: Logistic Regression, Support Vector Machine (SVM), and Random Forest
- Interactive Streamlit web application for predictions
- Exploratory Data Analysis (EDA) with visualizations
- Model performance comparison dashboard

## Project Structure
- `TCA_mahiri.ipynb`: Jupyter notebook with complete project implementation
- `streamlit_app.py`: Interactive Streamlit web application
- `save_models.py`: Script to train and save the ML models
- `Customer Churn.csv`: Dataset used for training and testing
- Model files: Saved trained models and preprocessing objects
- `churn_analysis_project.zip`: Complete project archive

## How to Run

### Prerequisites
- Python 3.7+
- Required packages: streamlit, pandas, numpy, scikit-learn, matplotlib, seaborn, joblib

### Installation
1. Download and extract this project package
2. Install required packages: `pip install streamlit pandas numpy scikit-learn matplotlib seaborn joblib`
3. Run the save_models.py script to train and save the models: `python save_models.py`
4. Run the Streamlit app: `streamlit run streamlit_app.py`

### Usage
1. After running the app, access it through your web browser using the URL displayed in the terminal (typically http://localhost:8501 unless that port is occupied)
2. Navigate to the Prediction page to enter customer information and predict churn risk
3. View model performance comparison on the Model Comparison page
4. Explore the dataset with visualizations on the EDA page

## Team Members
- Pathan Mahir
- Patwa Faizan
- Vasava Jayesh

## Academic Details
- Course: BCA Semester 6
- Subject: Data Analytics using Python
- Dataset: Telco Customer Churn (downloaded from Kaggle)

## Algorithms Used
- **Logistic Regression**: Interpretable linear model ideal for binary classification tasks
- **Support Vector Machine**: Effective for complex patterns and high-dimensional spaces
- **Random Forest**: Ensemble method combining multiple decision trees with feature importance rankings