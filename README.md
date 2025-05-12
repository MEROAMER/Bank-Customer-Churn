# Bank Customer Churn Prediction

A machine learning project that predicts whether a customer will leave the bank. Built for data analysts, data scientists, and product teams who want to identify churn risk early and take action.

## Features

- End-to-end pipeline: EDA, preprocessing, modeling, evaluation
- Streamlit web app for real-time churn prediction
- Pickle-based deployment with saved model, scaler, and encoders

## Files

- `Bank Churn Prediction.ipynb`: Contains all EDA, preprocessing, model training, and evaluation
- `main.py`: Streamlit application for user-friendly churn prediction
- `churn_model.pkl`: Pickled model containing classifier, scaler, encoders, and feature list
- `Churn_Modelling.csv`: Original dataset used for training and evaluation
- `README.md`: Documentation and usage instructions

## Installation

Clone the repo:
```bash
git clone https://github.com/MEROAMER/Bank-Customer-Churn.git
cd Bank-Customer-Churn
```

Install the requirements:
```bash
pip install -r requirements.txt
```

Run the Streamlit app:
```bash
streamlit run main.py
```
