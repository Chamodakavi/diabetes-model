import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load model and dataset
model = joblib.load('main.pkl')  # trained without SMOTE step in pipeline
df = pd.read_csv('diabetes_dataset.csv')

# Define the same features used during training
selected_features = [
    'Age', 'BMI', 'Glucose', 'DiabetesPedigreeFunction',
    'BMI_Category_Underweight', 'BMI_Category_Normal', 'BMI_Category_Overweight', 'BMI_Category_Obese',
    'Glucose_Category_Good', 'Glucose_Category_Broad', 'Glucose_Category_Bad'
]

# Title and description
st.title("Diabetes Prediction App")
st.write("""
    This app predicts whether a person has diabetes based on input features.
    The model uses historical medical data such as glucose levels, age, BMI, etc.
""")

# Sidebar navigation
sidebar = st.sidebar.radio("Select a Section", 
                            ("Data Exploration", "Visualization", "Model Prediction", "Model Performance"))

# -------------------
# Data Exploration
# -------------------
if sidebar == "Data Exploration":
    st.subheader("Dataset Overview")
    st.write(f"Dataset Shape: {df.shape}")
    st.write("Columns: ", list(df.columns))
    st.write("Data Types: ", df.dtypes)
    
    st.subheader("Sample Data")
    st.dataframe(df.head())
    
    st.subheader("Filter Data")
    selected_column = st.selectbox("Select a column to filter", df.columns)
    min_val, max_val = st.slider("Select range of values",
                                 min_value=float(df[selected_column].min()), 
                                 max_value=float(df[selected_column].max()),
                                 value=(float(df[selected_column].min()), float(df[selected_column].max())))
    filtered_df = df[(df[selected_column] >= min_val) & (df[selected_column] <= max_val)]
    st.dataframe(filtered_df)

# -------------------
# Visualization
# -------------------
if sidebar == "Visualization":
    st.subheader("Data Visualizations")
    
    st.subheader("Glucose Level Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Glucose'], kde=True, ax=ax)
    st.pyplot(fig)
    
    st.subheader("BMI vs Age")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Age', y='BMI', data=df, ax=ax)
    st.pyplot(fig)
    
    st.subheader("Correlation Heatmap")
    corr = df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# -------------------
# Model Prediction
# -------------------
if sidebar == "Model Prediction":
    st.subheader("Enter Feature Values to Predict Diabetes")
    
    age = st.number_input("Age", min_value=20, max_value=100, value=30)
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
    glucose = st.number_input("Glucose", min_value=0.0, max_value=200.0, value=100.0)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)

    # BMI category flags
    bmi_underweight = 0
    bmi_normal = 0
    bmi_overweight = 0
    bmi_obese = 0

    if bmi < 18.5:
        bmi_underweight = 1
    elif bmi < 25:
        bmi_normal = 1
    elif bmi < 30:
        bmi_overweight = 1
    else:
        bmi_obese = 1

    # Glucose category flags
    glucose_good = 0
    glucose_broad = 0
    glucose_bad = 0

    if glucose < 90:
        glucose_good = 1
    elif glucose < 140:
        glucose_broad = 1
    else:
        glucose_bad = 1

    # Build input DataFrame with same feature order as training
    input_df = pd.DataFrame([[age, bmi, glucose, diabetes_pedigree_function,
                              bmi_underweight,
                              bmi_normal,
                              bmi_overweight,
                              bmi_obese,
                              glucose_good,
                              glucose_broad,
                              glucose_bad]],
                            columns=selected_features)

    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.write("**Prediction:** Diabetes")
        else:
            st.write("**Prediction:** No Diabetes")

        st.write(f"Prediction Confidence: {probability * 100:.2f}%")

# -------------------
# Model Performance
# -------------------
if sidebar == "Model Performance":
    st.subheader("Model Performance Evaluation")

    # Create the BMI and Glucose category dummy columns first
    df['BMI_Category_Underweight'] = (df['BMI'] < 18.5).astype(int)
    df['BMI_Category_Normal'] = ((df['BMI'] >= 18.5) & (df['BMI'] < 25)).astype(int)
    df['BMI_Category_Overweight'] = ((df['BMI'] >= 25) & (df['BMI'] < 30)).astype(int)
    df['BMI_Category_Obese'] = (df['BMI'] >= 30).astype(int)

    df['Glucose_Category_Good'] = (df['Glucose'] < 90).astype(int)
    df['Glucose_Category_Broad'] = ((df['Glucose'] >= 90) & (df['Glucose'] < 140)).astype(int)
    df['Glucose_Category_Bad'] = (df['Glucose'] >= 140).astype(int)

    # Now select features for evaluation
    df_eval = df[selected_features].copy()
    y_test = df['Outcome']

    y_pred = model.predict(df_eval)
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)
