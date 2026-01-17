import streamlit as st

st.set_page_config(page_title="News Category Classifier", layout="wide")

if "api_base" not in st.session_state:
    st.session_state["api_base"] = "http://127.0.0.1:8000"

st.session_state["api_base"] = st.sidebar.text_input(
    "API Base URL",
    st.session_state["api_base"]
)

st.title("News Headline and Category Dashboard")
