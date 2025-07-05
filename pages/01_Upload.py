from pathlib import Path
import streamlit as st
import pandas as pd

# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Trajectory Estimation",
    page_icon="📡",
    layout="centered",
    initial_sidebar_state="collapsed",  # sidebar closed on this page
)

# -------------------------------------------------------------------
# 1. Title + instructions
# -------------------------------------------------------------------
st.title("📡 Trajectory Estimation")
st.subheader("Upload base‑station and pseudorange CSV files")

st.markdown(
    """
    **How to use this demo**

    1. Prepare **two CSV files**:
       * **Base‑station file** – `Title, Latitude, Longitude, Height`
       * **Pseudorange file**  – one column per base‑station (`BS1, BS2, …`) containing *metres* for every epoch.
    2. *(Optional)* A **true trajectory** CSV with `Latitude,Longitude` columns lets you compare estimates against ground truth.
    3. Upload the files with the widgets below.
    4. Click **Continue ▶︎** to choose an estimator and visualise results.
    """
)

# -------------------------------------------------------------------
# 2. Template download buttons
# -------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "DemoData"
bs_sample   = DATA_DIR / "base_stations.csv"
pr_sample   = DATA_DIR / "pseudoranges.csv"
truth_sample = DATA_DIR / "true-trajectory.csv"

cols = st.columns(3)
if bs_sample.exists():
    cols[0].download_button("⬇️ Example base‑stations", bs_sample.read_bytes(), "base_stations_template.csv", mime="text/csv")
if pr_sample.exists():
    cols[1].download_button("⬇️ Example pseudoranges", pr_sample.read_bytes(), "pseudoranges_template.csv", mime="text/csv")
if truth_sample.exists():
    cols[2].download_button("⬇️ Example true track", truth_sample.read_bytes(), "true_track_template.csv", mime="text/csv")

# -------------------------------------------------------------------
# 3. File‑upload widgets
# -------------------------------------------------------------------
uploaded_bs    = st.file_uploader("Upload base‑station CSV", type="csv", key="bs_csv")
uploaded_pr    = st.file_uploader("Upload pseudoranges CSV",  type="csv", key="pr_csv")
uploaded_truth = st.file_uploader("(Optional) Upload true trajectory CSV", type="csv", key="truth_csv")

with st.expander("🔍 CSV column layout", expanded=False):
    st.markdown(
        """
        **Base‑station file**
        ```csv
        Title,Latitude,Longitude,Height
        BS1,45.3435,9.0102,175.3
        BS2,45.3435,9.0149,175.3
        ```
        **Pseudorange file** (example with 8 BS)
        ```csv
        BS1,BS2,BS3,BS4,BS5,BS6,BS7,BS8
        371.946,376.128, …
        372.055,376.237, …
        ```
        **True trajectory**
        ```csv
        Latitude,Longitude
        45.3436,9.0110
        45.3437,9.0111
        ```
        All units are **metres** for heights / pseudoranges and **degrees** for lat/lon.
        """
    )

# -------------------------------------------------------------------
# 4. Load / validate / cache & continue
# -------------------------------------------------------------------
if uploaded_bs and uploaded_pr:
    try:
        BS = pd.read_csv(uploaded_bs).to_numpy()
        P  = pd.read_csv(uploaded_pr).to_numpy()
    except Exception as e:
        st.error(f"❌ Failed to read one of the CSVs: {e}")
        st.stop()

    st.session_state["BS"] = BS
    st.session_state["P"]  = P

    # Optional truth
    if uploaded_truth is not None:
        try:
            truth_arr = pd.read_csv(uploaded_truth)[["Latitude", "Longitude"]].to_numpy()
            st.session_state["TRUTH"] = truth_arr
            st.success(
                f"✅ Loaded {BS.shape[0]} BS, {P.shape[0]} epochs, true track with {truth_arr.shape[0]} points."
            )
        except Exception as e:
            st.warning(f"True track ignored – could not read file: {e}")
            st.session_state["TRUTH"] = None
    else:
        st.session_state["TRUTH"] = None
        st.success(f"✅ Loaded {BS.shape[0]} base stations and {P.shape[0]} epochs.")

    if st.button("Continue ▶︎", type="primary"):
        try:
            st.switch_page("pages/02_Estimation.py")
        except AttributeError:
            st.error("Streamlit ≥1.25 required for multipage support.")
else:
    st.info("Please upload **both** base‑station and pseudorange CSV files to continue.")
