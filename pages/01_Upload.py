from pathlib import Path
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Trajectory Estimation",
    page_icon="📡",
    layout="centered",
    initial_sidebar_state="collapsed",  # keep sidebar closed on this page
)

st.title("📡 Trajectory Estimation")



st.markdown(
    """
    **How to use this app**

    1. Prepare **two CSV files** using the column layouts below (WGS‑84 lat/lon in **degrees**, heights in **metres**):
       * **Base‑station file** – `Title, Latitude, Longitude, Height`  
       * **Pseudorange file**  – one column per base‑station (`BS1, BS2, …`) containing *metres* for every epoch.
    2. Upload the files with the widgets below.
    3. Pick an estimator in the sidebar (pure Least‑Squares, Least-Squares + Kalman filter, or Extended Kalman filter), tune the noise sigmas, and press **Run**.

    > **Need something to test with first?** Download ready‑made example files:
    """
)

SAMPLE_DIR = Path(__file__).parent.parent
bs_sample = SAMPLE_DIR / "base_stations.csv"
pr_sample = SAMPLE_DIR / "pseudoranges.csv"

cols_dl = st.columns(2)
if bs_sample.exists():
    cols_dl[0].download_button(
        "⬇️ Example base‑stations CSV",
        data=bs_sample.read_bytes(),
        file_name="base_stations_template.csv",
        mime="text/csv",
    )
if pr_sample.exists():
    cols_dl[1].download_button(
        "⬇️ Example pseudoranges CSV",
        data=pr_sample.read_bytes(),
        file_name="pseudoranges_template.csv",
        mime="text/csv",
    )
else:
    st.warning("Example pseudoranges CSV not found. Please check the sample_data directory.")

uploaded_pr = st.file_uploader("Upload pseudoranges CSV", type="csv", key="pr")
uploaded_bs = st.file_uploader("Upload base-station CSV", type="csv", key="bs")

with st.expander("🔍 CSV layout details", expanded=False):
    st.markdown(
        """
        **Base‑station file**
        ```csv
        Title,Latitude,Longitude,Height
        BS1,45.3435,9.0102,175.3
        BS2,45.3435,9.0149,175.3
        …
        ```
        **Pseudorange file** (8 BS example)
        ```csv
        BS1,BS2,BS3,BS4,BS5,BS6,BS7,BS8
        371.946,376.128, …
        372.055,376.237, …
        …
        ```
        All values are *metres*.  One row per epoch.
        """
    )


if uploaded_pr and uploaded_bs:
    P = pd.read_csv(uploaded_pr).to_numpy()
    BS = pd.read_csv(uploaded_bs).to_numpy()
    try:
        st.switch_page("pages/02_Estimation.py")
    except AttributeError:
        st.error(
            "Something went wrong! "
            "Please make sure you are running Streamlit 1.25+.\n"
        )
else:
    st.warning("Please upload both pseudoranges and base-station CSV files to begin.")
    st.stop() 