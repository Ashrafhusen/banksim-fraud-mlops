import streamlit as st
import pandas as pd
from joblib import load
import shap
import matplotlib.pyplot as plt

model, feature_names = load("models/model.joblib")

st.title ("Fraud Detection")

st.header("Transaction Details")

step = st.slider("Step(hour)", 1,744, 1)
amount = st.number_input("Amount", min_value  = 0.01, step = 0.01)
amount_log = st.number_input("Amount(Log)", value = 0.0)
hour_of_day = step % 24


customer = st.number_input("Customer ID (encoded)", 0, 5000, 100)
merchant = st.number_input("Merchant ID (encoded)", 0, 500, 100)
category = st.number_input("Category (encoded)", 0, 15, 5)
gender = st.number_input("Gender (encoded)", 0, 1, 0)
age = st.number_input("Age (0–5 group)", 0, 5, 3)

if st.button("Predict Fraud"):
    input_df = pd.DataFrame([{
        "step": step,
        "amount": amount,
        "amount_log": amount_log,
        "hour_of_day": hour_of_day,
        "customer": customer,
        "merchant": merchant,
        "category": category,
        "gender": gender,
        "age": age
    }])

    input_df = input_df[feature_names]

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]


    st.subheader(f"Prediction: {'FRAUD' if pred == 1 else 'LEGITIMATE'}")
    st.metric("Fraud Probability", f"{proba:.2%}")

    st.subheader("Prediction Explanation (SHAP)")

    booster = model.get_booster()
    booster.feature_names = input_df.columns.tolist()


    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(input_df)


    plt.title("SHAP Waterfall Plot")
    shap.plots.waterfall(shap.Explanation(values=shap_values[0],
                                          base_values=explainer.expected_value,
                                          data=input_df.iloc[0]), show=False)
    st.pyplot(plt)



    