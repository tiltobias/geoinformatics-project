import streamlit as st

st.set_page_config(
    page_title="Trajectory Estimation",
    page_icon="📡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

try:
    st.switch_page("pages/01_Upload.py")
except AttributeError:
    st.error(
        "Something went wrong! "
        "Please make sure you are running Streamlit 1.25+.\n"
    )