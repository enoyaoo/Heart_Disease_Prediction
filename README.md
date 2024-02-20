# Heart Attack Prediction
by - Chetan Sarda, Zixuan Zhu, Yinong Yao, Randeep Singh

## Overview
This project aims to predict the likelihood of heart attacks based on various health indicators using machine learning techniques. The goal is to assist healthcare professionals in identifying high-risk individuals for preventive measures.

## Installation
To set up the project, install the required packages:

pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn fairlearn shap

## Data
The dataset contains health-related features such as age, gender, BMI, smoking status, physical activity, and more. It is preprocessed to handle missing values, encode categorical variables, and normalize numerical features. Link: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease

## Feature Selection
Features are selected using Recursive Feature Elimination with Cross-Validation (RFECV) to identify the most relevant predictors for heart attack risk.

## Model Selection
Several classifiers are evaluated, including Logistic Regression, K-Nearest Neighbors, Decision Tree, Support Vector Machine, Gaussian Naive Bayes, and Random Forest. The models are tuned using hyperparameter optimization to enhance performance.

## Final Model
The Random Forest Classifier is chosen as the final model based on its balance between accuracy and generalization.

## Fairness Assessment
The model's fairness is assessed using the Fairlearn library to ensure that predictions are equitable across different demographic groups.

## Risk Categorization
Predictions are categorized into different risk levels (Low, Medium, High) to aid in prioritizing medical interventions.

## Usage
To run the project, execute the Jupyter notebook that includes data preprocessing, model training, and evaluation.
