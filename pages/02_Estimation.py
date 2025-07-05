import streamlit as st
from pathlib import Path
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium

# ── project modules (adjust imports to your structure) ──────────────
# from src.lsm    import solve   as run_lsm      # Least‑Squares
# from src.kalman import smooth  as run_kf       # LS + Kalman smooth
# from src.ekf    import estimate as run_ekf     # Extended Kalman
# from src.coords import lc_to_gg                # LC → GG converter

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Trajectory Estimation",
    page_icon="📡",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("📡 Trajectory Estimation")
st.subheader("Estimate and compare trajectory from pseudorange data")

# -------------------------------------------------------------------
# 1. Load uploaded data
# -------------------------------------------------------------------
if "BS" not in st.session_state or "P" not in st.session_state:
    st.error("No data found – please upload files on the previous page.")
    st.stop()

BS = st.session_state["BS"]       # shape (n_stations, 4)
P  = st.session_state["P"]        # shape (n_epochs, n_stations)
TRUTH = st.session_state.get("TRUTH")  # optional array (m,2)

n_epochs, n_stations = P.shape
st.sidebar.success(f"Data: {n_stations} stations, {n_epochs} epochs loaded")

# -------------------------------------------------------------------
# 2. Sidebar controls
# -------------------------------------------------------------------
st.sidebar.title("Choose estimator")

# Least‑Squares only
with st.sidebar.expander("Least‑Squares"):
    lsm_btn = st.button("▶ Run Least‑Squares")

# Least‑Squares + Kalman
with st.sidebar.expander("Least‑Squares + Kalman"):
    q_pos = st.number_input("Process σ‑pos [m]",   0.0, 20.0, 0.5, 0.1, key="kf_sigma_pos")
    q_vel = st.number_input("Process σ‑vel [m/s]", 0.0, 10.0, 0.2, 0.1, key="kf_sigma_vel")
    kf_btn = st.button("▶ Run LS + Kalman")

# Extended Kalman
with st.sidebar.expander("Extended Kalman Filter"):
    ekf_sigma_pos = st.number_input("σ‑pos [m]",      0.0, 20.0, 0.5, 0.1, key="ekf_sigma_pos")
    ekf_sigma_vel = st.number_input("σ‑vel [m/s]",    0.0, 10.0, 0.2, 0.1, key="ekf_sigma_vel")
    ekf_sigma_clk = st.number_input("σ‑clock [ns]",   0.0, 10.0, 0.5, 0.1, key="ekf_sigma_clk")
    ekf_btn = st.button("▶ Run Extended Kalman")

# Layers toggles grouped in sidebar
with st.sidebar.expander("Map layers", expanded=True):
    show_truth = st.checkbox("True trajectory", value=(TRUTH is not None))
    show_bs = st.checkbox("Base‑stations", value=True)

st.markdown("&nbsp;")  # spacer

# -------------------------------------------------------------------
# 3. Run Least‑Squares
# -------------------------------------------------------------------
if lsm_btn:
    with st.spinner("Running Least‑Squares …"):
        lsm_lc = run_lsm(P, BS)
        lsm_gg = lc_to_gg(lsm_lc, BS)
    st.session_state["lsm_sol"] = {"lc": lsm_lc, "gg": lsm_gg}
    st.success("Least‑Squares complete.")

# -------------------------------------------------------------------
# 4. Run Kalman smoother
# -------------------------------------------------------------------
if kf_btn and "lsm_sol" in st.session_state:
    with st.spinner("Running Kalman smoothing …"):
        kf_lc = run_kf(st.session_state["lsm_sol"]["lc"], q_pos, q_vel)
        kf_gg = lc_to_gg(kf_lc, BS)
    st.session_state["kf_sol"] = {"lc": kf_lc, "gg": kf_gg}
    st.success("Kalman smoothing complete.")

# -------------------------------------------------------------------
# 5. Run Extended Kalman
# -------------------------------------------------------------------
if ekf_btn:
    with st.spinner("Running Extended Kalman …"):
        ekf_lc = run_ekf(P, BS, {
            "sigma_pos": ekf_sigma_pos,
            "sigma_vel": ekf_sigma_vel,
            "sigma_clk": ekf_sigma_clk,
        })
        ekf_gg = lc_to_gg(ekf_lc, BS)
    st.session_state["ekf_sol"] = {"lc": ekf_lc, "gg": ekf_gg}
    st.success("Extended Kalman complete.")

# -------------------------------------------------------------------
# 6. Map visualisation
# -------------------------------------------------------------------
cols = st.columns(4)
show_lsm = cols[0].checkbox("Least‑Squares",    value=("lsm_sol" in st.session_state))
show_kf  = cols[1].checkbox("Kalman smoothed",  value=("kf_sol" in st.session_state))
show_ekf = cols[2].checkbox("Extended Kalman",  value=("ekf_sol" in st.session_state))
st.markdown("&nbsp;")  # separate map toggles from layers

# Determine map center
if show_truth and TRUTH is not None:
    center_lat, center_lon = TRUTH[0]
elif "lsm_sol" in st.session_state:
    center_lat, center_lon = st.session_state["lsm_sol"]["gg"][0]
else:
    center_lat, center_lon = BS[:,1].mean(), BS[:,2].mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=17)

# Plot ground-truth
if show_truth and TRUTH is not None:
    folium.PolyLine(TRUTH.tolist(), color="green", weight=3, tooltip="Ground-truth").add_to(m)

# Plot base-stations
if show_bs:
    for title, lat, lon, h in BS:
        folium.Marker([lat, lon], popup=title, icon=folium.Icon(color="black",icon="tower")).add_to(m)

# Plot LSM
if show_lsm and "lsm_sol" in st.session_state:
    for lat, lon in st.session_state["lsm_sol"]["gg"][:, :2]:
        folium.CircleMarker([lat, lon], radius=2, color="blue", fill=True).add_to(m)

# Plot KF
if show_kf and "kf_sol" in st.session_state:
    folium.PolyLine(st.session_state["kf_sol"]["gg"][:, :2].tolist(), color="red", weight=3, tooltip="Kalman").add_to(m)

# Plot EKF
if show_ekf and "ekf_sol" in st.session_state:
    folium.PolyLine(st.session_state["ekf_sol"]["gg"][:, :2].tolist(), color="orange", weight=3, tooltip="EKF").add_to(m)

folium.LayerControl().add_to(m)
st_folium(m, width=800, key="trajectory_map")