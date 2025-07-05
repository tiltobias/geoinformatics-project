import streamlit as st
import numpy as np
import folium
from streamlit_folium import st_folium

# ── project modules ────────────────────────────────────────────────
from modules import (
    create_origin,
    transform_GG_to_LC,
    transform_LC_to_GG,
    calculate_LSM,
    calculate_kalman,
    calculate_ex_kalman,
)

# aliases
run_lsm   = calculate_LSM
run_kf    = calculate_kalman
run_ex_kf = calculate_ex_kalman

# ------------------------------------------------------------------
# Streamlit page config
# ------------------------------------------------------------------
st.set_page_config(page_title="Trajectory Estimation",
                   page_icon="📡",
                   layout="centered",
                   initial_sidebar_state="expanded")

st.title("📡 Trajectory Estimation")
st.subheader("Estimate and compare trajectory from pseudorange data")

# ------------------------------------------------------------------
# 1. Data prepared on the Upload page
# ------------------------------------------------------------------
if "BS" not in st.session_state or "P" not in st.session_state:
    st.error("No data found – please upload files on the previous page.")
    st.stop()

titles  = st.session_state["BS"][:, 0]
bs_gg   = st.session_state["BS"][:, 1:4].astype(float)    # (n,3)
P       = st.session_state["P"].astype(float)             # (epochs, n)
TRUTH   = st.session_state.get("TRUTH")                   # optional

n_epochs, n_stations = P.shape
st.sidebar.success(f"Data loaded – {n_stations} stations, {n_epochs} epochs")

# ------------------------------------------------------------------
# 2. Local Cartesian system
# ------------------------------------------------------------------
origin_GG, _ = create_origin(bs_gg)               # lat-lon-h (deg)
bs_lc        = transform_GG_to_LC(bs_gg, origin_GG)

# ------------------------------------------------------------------
# 3. Sidebar – estimator options
# ------------------------------------------------------------------
st.sidebar.title("Choose estimator")

with st.sidebar.expander("Least-Squares"):
    lsm_btn = st.button("▶ Run Least-Squares")

with st.sidebar.expander("Least-Squares + Kalman"):
    q_pos = st.number_input("Observation σ [m]", 0.0, 20.0, 3.0, 0.5)
    q_vel = st.number_input("Model σ error [m]", 0.0, 10.0, 0.5, 0.1)
    kf_btn = st.button("▶ Run LS + Kalman")

with st.sidebar.expander("Extended Kalman Filter"):
    # cadence
    if "t" in st.session_state and len(st.session_state["t"]) > 1:
        dt_default = float(np.median(np.diff(st.session_state["t"])))
    else:
        dt_default = 1.0
    dt_ekf   = st.number_input("Time step dt [s]", 0.01, 10.0, dt_default, 0.05)
    r_pr_ekf = st.number_input("Meas. σ pseudorange [m]", 0.0, 20.0, 3.0, 0.5)
    sigma_pos = st.number_input("Process σ position [m]",   0.0, 20.0, 1.0, 0.1)
    sigma_vel = st.number_input("Process σ velocity [m/s]", 0.0, 10.0, 0.5, 0.1)
    sigma_dt  = st.number_input("Clock offset σ [ns]",       0.0, 10.0, 1.0, 0.5)
    ekf_btn   = st.button("▶ Run Extended Kalman Filter")

with st.sidebar.expander("Map layers", expanded=True):
    show_truth = st.checkbox("True trajectory", value=(TRUTH is not None), disabled=(TRUTH is None))
    show_bs    = st.checkbox("Base-stations",   value=True)

st.markdown("&nbsp;")

# ------------------------------------------------------------------
# 0. Widget-state defaults (only once)
# ------------------------------------------------------------------
for flag in ("show_lsm", "show_kf", "show_ekf"):
    st.session_state.setdefault(flag, False)

# ------------------------------------------------------------------
# 4. Run Least-Squares (explicit button)
# ------------------------------------------------------------------
if lsm_btn:
    with st.spinner("Running Least-Squares …"):
        lsm_lc = run_lsm(P, bs_lc)
        lsm_gg = transform_LC_to_GG(lsm_lc[:, :3], origin_GG)
    st.session_state["lsm_sol"]  = {"lc": lsm_lc, "gg": lsm_gg}
    st.session_state["show_lsm"] = True
    st.success("Least-Squares finished.")

# ------------------------------------------------------------------
# 5. Run Kalman smoother  (LS → KF)
# ------------------------------------------------------------------
if kf_btn:
    if "lsm_sol" not in st.session_state:            # silent LS
        lsm_lc = run_lsm(P, bs_lc)
        lsm_gg = transform_LC_to_GG(lsm_lc[:, :3], origin_GG)
        st.session_state["lsm_sol"] = {"lc": lsm_lc, "gg": lsm_gg}
        # do NOT touch show_lsm here

    with st.spinner("Running LS + Kalman …"):
        kf_state = run_kf(
            st.session_state["lsm_sol"]["lc"][:, :2],
            sigma_obs   = q_pos,
            sigma_error = q_vel,
        )
        kf_EN = kf_state[:, :2, 0]
        kf_lc = np.column_stack([kf_EN, np.zeros(len(kf_EN))])
        kf_gg = transform_LC_to_GG(kf_lc, origin_GG)

    st.session_state["kf_sol"]  = {"lc": kf_lc, "gg": kf_gg}
    st.session_state["show_kf"] = True
    st.success("Kalman finished.")

# ------------------------------------------------------------------
# 6. Run Extended-Kalman filter
# ------------------------------------------------------------------
if ekf_btn:
    with st.spinner("Running Extended Kalman Filter …"):
        ekf_states, _ = run_ex_kf(
            P                 = P,
            base_stations     = bs_lc,
            dt                = dt_ekf,
            sigma_dt          = sigma_dt * 1e-9,
            sigma_pos         = sigma_pos,
            sigma_vel         = sigma_vel,
            sigma_pseudorange = r_pr_ekf
        )
        ekf_lc = np.array(ekf_states)[:, :3]  # (epochs, E,N,U)
        ekf_gg = transform_LC_to_GG(ekf_lc, origin_GG)

    st.session_state["ekf_sol"]  = {"lc": ekf_lc, "gg": ekf_gg}
    st.session_state["show_ekf"] = True
    st.success("Extended Kalman finished.")

# ------------------------------------------------------------------
# 7. Layer check-boxes (stateful)
# ------------------------------------------------------------------
cols = st.columns(3)
show_lsm = cols[0].checkbox("Least-Squares",          key="show_lsm", disabled=("lsm_sol" not in st.session_state))
show_kf  = cols[1].checkbox("Least-Squares + Kalman", key="show_kf", disabled=("kf_sol" not in st.session_state))
show_ekf = cols[2].checkbox("Extended Kalman Filter", key="show_ekf", disabled=("ekf_sol" not in st.session_state))

# ------------------------------------------------------------------
# 8. Decide map centre
# ------------------------------------------------------------------
if show_truth and TRUTH is not None:
    center_lat, center_lon = TRUTH[0]
elif show_ekf and "ekf_sol" in st.session_state:
    center_lat, center_lon = st.session_state["ekf_sol"]["gg"][0, :2]
elif show_kf and "kf_sol" in st.session_state:
    center_lat, center_lon = st.session_state["kf_sol"]["gg"][0, :2]
elif show_lsm and "lsm_sol" in st.session_state:
    center_lat, center_lon = st.session_state["lsm_sol"]["gg"][0, :2]
else:
    center_lat, center_lon = bs_gg[:, 0].mean(), bs_gg[:, 1].mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=17)

# ------------------------------------------------------------------
# 9. Draw layers
# ------------------------------------------------------------------
if show_truth and TRUTH is not None:
    folium.PolyLine(TRUTH.tolist(), color="green", weight=3,
                    tooltip="Ground-truth").add_to(m)

if show_bs:
    for title, (lat, lon, _) in zip(titles, bs_gg):
        folium.Marker([lat, lon], popup=title,
                      icon=folium.Icon(color="black", icon="tower")).add_to(m)

if show_lsm and "lsm_sol" in st.session_state:
    folium.PolyLine(st.session_state["lsm_sol"]["gg"][:, :2].tolist(),
                    color="blue", weight=3,
                    tooltip="Least-Squares").add_to(m)

if show_kf and "kf_sol" in st.session_state:
    folium.PolyLine(st.session_state["kf_sol"]["gg"][:, :2].tolist(),
                    color="red", weight=3,
                    tooltip="LS + Kalman").add_to(m)

if show_ekf and "ekf_sol" in st.session_state:
    folium.PolyLine(st.session_state["ekf_sol"]["gg"][:, :2].tolist(),
                    color="purple", weight=3,
                    tooltip="Extended Kalman").add_to(m)

folium.LayerControl().add_to(m)
st_folium(m, width=800, key="trajectory_map")
