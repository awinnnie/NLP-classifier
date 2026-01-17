import streamlit as st
import requests

if "api_base" not in st.session_state:
    st.session_state["api_base"] = "http://127.0.0.1:8000"
    
API_BASE = st.session_state["api_base"]

st.header("Data Manipulation")


# Add Headline

st.header("1. Add a New Headline")

new_headline = st.text_input(
    "New headline text",
    key="add_headline_text"
)

if st.button("Add Headline", key="add_btn"):
    if new_headline.strip() == "":
        st.warning("Headline cannot be empty.")
    else:
        try:
            r = requests.post(
                f"{API_BASE}/add_headline",
                json={"text": new_headline},
                timeout=10
            )

            r.raise_for_status()
            st.success("Headline added successfully!")
            st.json(r.json())
        except Exception as e:
            st.error(f"Add failed: {e}")

st.divider()

# Update headline
st.header("2. Update Headline")

upd_id = st.number_input(
    "Headline ID",
    min_value=0,
    format="%d",
    key="update_id"
)

upd_text = st.text_input(
    "Updated headline text",
    key="update_text"
)

if st.button("Update Headline", key="update_btn"):
    try:
        payload = {"id": int(upd_id), "text": upd_text}
        r = requests.put(
            f"{API_BASE}/update_headline",
            json=payload,
            timeout=10
        )
        r.raise_for_status()
        st.success("Headline updated!")
        st.json(r.json())
    except Exception as e:
        st.error(f"Update failed: {e}")

st.divider()

# Delete headline
st.header("3. Delete Headline")

del_id = st.number_input(
    "Headline ID to delete",
    min_value=0,
    format="%d",
    key="delete_id"
)

if st.button("Delete Headline", key="delete_btn"):
    try:
        r = requests.delete(
            f"{API_BASE}/delete_headline",
            json={"id": int(del_id)},
            timeout=10
        )
        r.raise_for_status()
        st.success("Headline deleted!")
        st.json(r.json())
    except Exception as e:
        st.error(f"Delete failed: {e}")
