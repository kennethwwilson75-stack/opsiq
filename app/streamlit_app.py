import streamlit as st

st.set_page_config(
    page_title="OpsIQ",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ OpsIQ — Industrial Fleet Intelligence")
st.write("App is running successfully on Streamlit Cloud")
st.success("Pipeline ready")

with st.sidebar:
    st.markdown("### OpsIQ")
    st.markdown("Multi-agent industrial fleet intelligence")
    page = st.radio(
        "Select view",
        ["Live Analysis", "Plant Dashboard", "Executive Report", "Chat Interface"]
    )

st.write(f"Selected: {page}")