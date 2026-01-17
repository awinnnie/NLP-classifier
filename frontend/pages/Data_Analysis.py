import streamlit as st
import requests
import pandas as pd

if "api_base" not in st.session_state:
    st.session_state["api_base"] = "http://127.0.0.1:8000"
    
API_BASE = st.session_state["api_base"]

st.header("Data Analysis")

CATEGORIES = [
    "POLITICS",
    "WELLNESS",
    "ENTERTAINMENT",
    "TRAVEL",
    "STYLE & BEAUTY",
    "PARENTING",
    "HEALTHY LIVING",
    "QUEER VOICES",
    "FOOD & DRINK",
    "BUSINESS",
    "COMEDY",
    "SPORTS",
    "BLACK VOICES",
    "HOME & LIVING",
    "PARENTS"
]

# Count Category
st.header("1. Count Headlines by Category")

category = st.selectbox(
    "Select category",
    CATEGORIES,
    key="count_category"
)

if st.button("Get Count"):
    try:
        r = requests.get(
            f"{API_BASE}/count/{category}",
            timeout=10
        )
        r.raise_for_status()
        st.info(f"Number of headlines: **{r.json()}**")
    except Exception as e:
        st.error(f"Count failed: {e}")

st.divider()

# Top 10 Headlines
st.header("2. Top 10 Headlines by Category")


top_cat = st.selectbox(
    "Select category",
    CATEGORIES,
    key="top10_category"
)

if st.button("Load Top 10"):
    try:
        r = requests.get(
            f"{API_BASE}/top10/{top_cat}",
            timeout=10
        )
        r.raise_for_status()
        data = r.json()

        df = pd.DataFrame(data["headlines"])
        st.subheader(f"Category: {data['category']}")
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load top 10: {e}")