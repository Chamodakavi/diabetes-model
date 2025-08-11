# Diabetes Prediction Model

## Overview

This repository contains a machine learning project for predicting the onset of diabetes based on various medical diagnostic measurements. The project includes data analysis, feature engineering, model training, and a web application for interactive predictions.

The final model is implemented in a Streamlit web application that allows users to input their medical data and receive a real-time prediction on whether they are likely to have diabetes.

## Features

- **Data Exploration & Visualization:** Interactive charts and data tables to explore the diabetes dataset.
- **Feature Engineering:** Creation of new categorical features from `BMI` and `Glucose` levels to improve model performance.
- **Model Training:** Explores and compares multiple classification models, including Logistic Regression, Random Forest, and XGBoost.
- **Class Imbalance Handling:** Uses Synthetic Minority Over-sampling Technique (SMOTE) to address the imbalanced nature of the dataset.
- **Interactive Web App:** A user-friendly interface built with Streamlit for making predictions and viewing model performance metrics.

## Dataset

The project uses the "Diabetes Dataset" which contains several medical predictor variables and one target variable, `Outcome`. The dataset includes features such as:

- `Pregnancies`: Number of times pregnant
- `Glucose`: Plasma glucose concentration
- `BloodPressure`: Diastolic blood pressure (mm Hg)
- `SkinThickness`: Triceps skin fold thickness (mm)
- `Insulin`: 2-Hour serum insulin (mu U/ml)
- `BMI`: Body mass index
- `DiabetesPedigreeFunction`: A function that scores the likelihood of diabetes based on family history
- `Age`: Age in years
- `Outcome`: Class variable (0 for no diabetes, 1 for diabetes)

The initial data preprocessing step involved replacing zero values in columns like `Glucose`, `BMI`, and `BloodPressure` with the median value of the respective column to handle missing or unrecorded data.

## Model Details

The final prediction model is a `RandomForestClassifier` integrated into a pipeline that includes:

1.  **SMOTE:** To oversample the minority class (patients with diabetes) and balance the training data.
2.  **StandardScaler:** To standardize features by removing the mean and scaling to unit variance.

Feature engineering was performed by creating categorical bins for `BMI` and `Glucose` and using one-hot encoding. The model was trained on the following selected features:

- `Age`
- `BMI`
- `Glucose`
- `DiabetesPedigreeFunction`
- `BMI_Category` (one-hot encoded)
- `Glucose_Category` (one-hot encoded)

The model is saved as `model2.pkl` for deployment in the Streamlit application.

## File Structure

```
.
├── Machine_Learning_Model_Deployment.ipynb   # Jupyter notebook for data analysis and model training
├── app.py                                    # The Streamlit web application script
├── diabetes_dataset.csv                      # The raw dataset
├── model2.pkl                                # The serialized trained model pipeline for the app
├── requirements.txt                          # Python dependencies for the project
└── model.pkl                                 # An alternative model pipeline saved during experimentation
```

## How to Run Locally

### Prerequisites

- Python 3.8+
- pip

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/chamodakavi/diabetes-model.git
    cd diabetes-model
    ```

2.  **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

Run the Streamlit app with the following command:

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your web browser to interact with the application.
