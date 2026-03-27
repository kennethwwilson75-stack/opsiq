import streamlit as st
st.secrets["ANTHROPIC_API_KEY"]

st.set_page_config(
    page_title="OpsIQ",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ OpsIQ Test")
st.write("Step 1 — basic rendering works")