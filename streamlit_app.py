import streamlit as st
import requests
import json
import os

st.set_page_config(page_title="Fraud Detection of Counterfeit Products", page_icon="🕵️", layout="centered")

st.title("🕵️ Fraud Detection Demo")
st.markdown("Enter the feature values below and click Predict to see if the instance is fraudulent.")

# Backend URL handling
# First try Streamlit secrets, then environment variable, finally default localhost
DEFAULT_API_URL = "http://localhost:8501/predict"

API_URL = os.getenv("API_URL", DEFAULT_API_URL)

with st.form("fraud_form"):
    vendor_score = st.number_input("Vendor Score (1.0 - 5.0)", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
    image_qty = st.number_input("Image Quantity (1 - 9)", min_value=1, max_value=20, value=5, step=1)
    site_age = st.number_input("Site Age (days, >=0)", min_value=0.0, value=1000.0, step=1.0)
    delivery_period = st.number_input("Delivery Period (days 1-50)", min_value=1, max_value=50, value=7, step=1)
    typo_count = st.number_input("Typo Count (>=0)", min_value=0, value=1, step=1)
    payment_options = st.number_input("Payment Options (1-5)", min_value=1, max_value=5, value=3, step=1)
    cost_usd = st.number_input("Cost USD (>0)", min_value=0.01, value=150.50, step=0.5, format="%.2f")

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "vendor_score": vendor_score,
        "image_qty": image_qty,
        "site_age": site_age,
        "delivery_period": delivery_period,
        "typo_count": typo_count,
        "payment_options": payment_options,
        "cost_usd": cost_usd
    }
    try:
        with st.spinner("Contacting model..."):
            resp = requests.post(API_URL, json=payload, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            pred_label = data.get("prediction_label", "?")
            confidence = data.get("confidence", 0.0)
            col1, col2 = st.columns(2)
            if pred_label.lower().startswith("fraud"):
                col1.error(f"Prediction: {pred_label}")
            else:
                col1.success(f"Prediction: {pred_label}")
            col2.info(f"Confidence: {confidence:.3f}")
            with st.expander("Raw Response"):
                st.json(data)
        else:
            st.error(f"API Error {resp.status_code}: {resp.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")


