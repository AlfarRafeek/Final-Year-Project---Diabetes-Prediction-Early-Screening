# Streamlit Dashboard for T2DM Prediction
# Run: streamlit run app.py
# Requirements: streamlit, pandas, joblib, plotly, matplotlib


# Install Streamlit if not already installed

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import matplotlib.pyplot as plt

# Load saved model and preprocessor
best_model = joblib.load('model/best_model.joblib')
preprocessor = joblib.load('model/preprocessor.joblib')

# Load comparison results
results_df = pd.read_csv('model_comparison.csv', index_col=0)

st.title('Type 2 Diabetes Early Screening Dashboard')
st.write('Developed using 271 anonymized records from Kattankudy Base Hospital, Sri Lanka.')
st.write('This dashboard allows model comparison, risk prediction, and key insights.')

# Section 1: Model Comparison
st.subheader('Model Performance Comparison')
st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'))

# Section 2: Interactive Prediction
st.subheader('Predict Your Diabetes Risk')
with st.form(key='prediction_form'):
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
    diet = st.selectbox('Diet', options=['healthy', 'unhealthy'])
    exercise_frequency = st.number_input('Exercise Frequency (days/week)', min_value=0, max_value=7, value=3)
    family_history = st.selectbox('Family History of Diabetes', options=['yes', 'no'])
    submit = st.form_submit_button('Predict')

if submit:
    # Create input DataFrame (match your columns)
    input_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'diet': [diet],
        'exercise_frequency': [exercise_frequency],
        'family_history': [family_history]
    })
    input_preprocessed = preprocessor.transform(input_data)
    prediction = best_model.predict(input_preprocessed)[0]
    prob = best_model.predict_proba(input_preprocessed)[0][1]
    risk_level = "High" if prediction == 1 else "Low"
    st.write(f'**Predicted Risk:** {risk_level} (Probability: {prob:.2%})')
    if risk_level == "High":
        st.warning('Consult a doctor for further screening.')

# Section 3: Visual Insights
st.subheader('Key Insights and Visualizations')

# Load and display correlation heatmap
st.image('correlation_heatmap.png', caption='Feature Correlations')

# Feature Importances (if available)
if 'feature_importances.html' in globals():  # Or check file existence
    with open('feature_importances.html', 'r') as f:
        html = f.read()
    st.components.v1.html(html, height=400)

# Additional Plot: Distribution of Age by Diabetes
# Assuming df is available; in practice, load df here if needed
# df = pd.read_csv('kattankudy_diabetes_data.csv')
# fig_age = px.histogram(df, x='age', color='diabetes', title='Age Distribution by Diabetes Status')
# st.plotly_chart(fig_age)

st.write('For more details, refer to the project documentation.')
