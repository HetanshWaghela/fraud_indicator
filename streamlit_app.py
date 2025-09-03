import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Fraud Detection of Counterfeit Products", page_icon="🕵️", layout="centered")

st.title("🕵️ Fraud Detection Demo")
st.markdown("Enter the feature values below and click Predict to see if the instance is fraudulent. (Local model mode)")

# Load model once (Option B: no Flask backend needed)
MODEL_PATH = "fraud_detection_model.pkl"
@st.cache_resource(show_spinner=True)
def load_model():
    try:
        model_loaded = joblib.load(MODEL_PATH)
        return model_loaded, None
    except Exception as e:
        return None, str(e)

model, model_err = load_model()
if model_err:
    st.error(f"Failed to load model: {model_err}")
    st.stop()

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
    input_row = {
        "vendor_score": float(vendor_score),
        "image_qty": int(image_qty),
        "site_age": float(site_age),
        "delivery_period": int(delivery_period),
        "typo_count": int(typo_count),
        "payment_options": int(payment_options),
        "cost_usd": float(cost_usd)
    }
    df = pd.DataFrame([input_row])
    try:
        if model is None:
            st.error("Model not loaded.")
            st.stop()
        with st.spinner("Scoring..."):
            pred = model.predict(df)[0]
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(df)[0]
                if len(probs) == 2:
                    prob_not, prob_fraud = probs
                else:
                    prob_fraud = probs[0]
                    prob_not = 1 - prob_fraud
            else:
                prob_fraud = np.nan
                prob_not = np.nan
        label = 'Fraud' if int(pred) == 1 else 'Not Fraud'
        confidence = float(max(prob_fraud, prob_not)) if not np.isnan(prob_fraud) else None
        col1, col2 = st.columns(2)
        if label == 'Fraud':
            col1.error(f"Prediction: {label}")
        else:
            col1.success(f"Prediction: {label}")
        if confidence is not None:
            col2.info(f"Confidence: {confidence:.3f}")
        with st.expander("Details"):
            st.write("Input:")
            st.json(input_row)
            if confidence is not None:
                st.write({"probability_not_fraud": float(prob_not), "probability_fraud": float(prob_fraud)})
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.caption("Loaded local model: " + MODEL_PATH)


