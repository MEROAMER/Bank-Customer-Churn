import streamlit as st
import pandas as pd
import pickle

# Load model, scaler, encoders, and feature list
with open('churn_model.pkl', 'rb') as f:
    model, scaler, label_encoders, features = pickle.load(f)

st.set_page_config(page_title="Bank Churn Prediction", layout="centered")
st.title("🏦 Bank Customer Churn Prediction")
st.markdown("Fill in the details below to predict if a customer is likely to churn.")

# ─── Input Form ─────────────────────────────────────────────
with st.form(key='churn_form'):
    credit_score = st.number_input("Credit Score", 300, 850, step=1)
    geography = st.selectbox("Geography", label_encoders['Geography'].classes_)
    gender = st.selectbox("Gender", ['Female', 'Male'])
    tenure = st.slider("Tenure (Years)", 0, 10, 3)
    age_group = st.selectbox("Age Group", label_encoders['AgeGroup'].classes_)
    balance = st.number_input("Balance", 0.0, 500000.0, step=100.0)
    num_products = st.slider("Number of Products", 1, 4, 1)
    has_card = st.radio("Has Credit Card?", [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
    is_active = st.radio("Is Active Member?", [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
    salary = st.number_input("Estimated Salary", 0.0, 300000.0, step=100.0)

    submit = st.form_submit_button(label="🔍 Predict")

# ─── Prediction Logic ───────────────────────────────────────
if submit:
    # Prepare input
    input_dict = {
        'CreditScore': credit_score,
        'Geography': label_encoders['Geography'].transform([geography])[0],
        'Gender': 1 if gender == 'Male' else 0,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': has_card,
        'IsActiveMember': is_active,
        'EstimatedSalary': salary,
        'AgeGroup': label_encoders['AgeGroup'].transform([age_group])[0]
    }

    input_df = pd.DataFrame([input_dict])

    # Scale numerical features
    scaled_features = ['CreditScore', 'Balance', 'EstimatedSalary']
    input_df[scaled_features] = scaler.transform(input_df[scaled_features])

    # Ensure correct feature order
    input_df = input_df[features]

    # Predict
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    # Display result
    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ This customer is likely to churn.\n\n**Churn Probability: {prob:.2%}**")
    else:
        st.success(f"✅ This customer is likely to stay.\n\n**Churn Probability: {prob:.2%}**")
