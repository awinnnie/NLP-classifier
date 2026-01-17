import streamlit as st
import requests

if "api_base" not in st.session_state:
    st.session_state["api_base"] = "http://127.0.0.1:8000"
    
API_BASE = st.session_state["api_base"]

st.header("Predict Category")

text = st.text_area("Enter a news headline")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter a headline.")
    else:
        try:
            r = requests.post(
                f"{API_BASE}/predict",
                json={"text": text},
                timeout=10
            )
            r.raise_for_status()
            result = r.json()
            st.success(f"Predicted category: **{result['prediction']}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")