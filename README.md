# Final_Year_Project - Diabetes_Prediction_Early_Screening
Risk prediction and early screening system - diabetes (Sri Lanka).

## Project Overview
It is an early screening project of Type 2 Diabetes Mellitus (T2DM) using machine learning and non-invasive questionnaire-based data on the Sri Lankan population. It is a system that is aimed at diagnosing the individuals at increased risk of developing diabetes before they can be diagnosed and provide awareness and prevention recommendations.

This essay is a Final Year Project done at the University of Westminster (IIT), Sri Lanka.

## Dataset Description
The primary data that was collected was collected using the structured survey via Google Forms and is made up of 271 anonymous responses of people of Kattankudy base hospital, Sri Lanka.

The dataset contains 21 attributes which include:
- Demographic factors (age, gender, occupation)
- Anthropometric (height, weight, BMI, waist circumference) measurements.
- Signs of the clinic (systolic and diastolic blood pressure).
- Habs (diet, physical activity, sleep duration)
- Symptom based (thirst, frequent urination, fatigue, blurred vision) variables.
- Family health and history.

The predictive set of features was reduced to those variables not related to diagnosis in order to avoid leakage of information.

## Machine Learning Models
Models of supervised machine learning that were tested and trained comprised:
- Logistic Regression
- Support Vector Machine (RBF Kernel)
- Random Forest Classifier
- Extra Trees Classifier
- HistGradient Boosting Classifier

Models training was performed on a reproducible preprocessing pipeline and the performance was compared on the basis of accuracy, recall, ROC-AUC and F1 score.

## Deployment
A web application based on Streamlit was used to deploy the trained model and allow the user to:
- Include information about individual health and lifestyle.
- Note an estimated risk of probability of diabetes.
- See descriptions on risk factors.
- Complete an awareness quiz
- Store a personalised diabetes awareness leaflet (PDF).

## Repository Structure
- streamlit_app.py - Streamlit application Python.  
- best_pipe.pkl - The trained machine learning model.  
- requirements.txt Python requirement packages.  
- README.md Documentation of the project.  
