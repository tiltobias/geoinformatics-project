# app.py ──────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from pathlib import Path

# ---- project modules ------------------------------------------------
from src.random_walk   import generate        as clk_walk
from src.pseudoranges  import make            as make_pseudoranges
from src.positioning   import lsm             # <-- your LS solver module
from src.kalman        import smooth          as kf_smooth
from src.coords        import lc_to_gg


C_LIGHT = 299_792_458.0

# ---- raw reference trajectory (lat/lon) -----------------------------
T1_PATH = Path(__file__).parent.parent / "KinematicData" / "T1.csv"
df_true = pd.read_csv(T1_PATH)                        # Latitude, Longitude, Height


# ──────────────────────────────────────────────────────────────────────
#  Sidebar – two independent sections
# ──────────────────────────────────────────────────────────────────────
st.sidebar.title("Trajectory estimations")

# -- 1.  Least-Squares block -----------------------------------------
with st.sidebar.expander("Least-Squares"):
    sigma_pr = st.number_input("Pseudorange noise σ [m]", 0.0, 50.0, 1.0, 0.1)
    rw_step  = st.number_input("Clock random-walk step [ns]", 0.0, 10.0, 0.5, 0.1)
    seed     = st.number_input("Random seed", 0, 2**32 - 1, 42, step=1)
    lsm_btn  = st.button("▶ Run Least-Squares")

# -- 2.  Kalman block -------------------------------------------------
with st.sidebar.expander("Least-Squares + Kalman filter"):
    q_pos  = st.number_input("Process noise σ-pos [m]",      0.0, 20.0, 1.0, 0.1)
    q_vel  = st.number_input("Process noise σ-vel [m/s]",    0.0, 10.0, 1.0, 0.1)
    r_meas = st.number_input("Measurement noise σ [m]",      0.0, 50.0, 5.0, 0.5)
    kf_btn = st.button("▶ Run Least-Squares + Kalman", disabled="lsm_res" not in st.session_state)

with st.sidebar.expander("Extended Kalman filter"):
    ex_btn  = st.button("▶ Run Extended Kalman")


st.write(" ")   # small vertical spacer


# ──────────────────────────────────────────────────────────────────────
#  1. Run / cache Least-Squares estimation
# ──────────────────────────────────────────────────────────────────────
if lsm_btn:
    with st.spinner("Generating clock offsets …"):
        n_epochs = len(df_true)
        clk_ns   = clk_walk(n_epochs, step_ns=rw_step, seed=seed)

    with st.spinner("Building pseudoranges …"):
        rho, ue_lc, bs_lc = make_pseudoranges(
            t1_csv=T1_PATH,
            bs_csv=Path(__file__).parent.parent / "base_stations.csv",
            clock_offset_ns=clk_ns,
            noise_sigma_m=sigma_pr,
            seed=seed,
        )

    with st.spinner("Running Least-Squares solver …"):
        lsm_lc = lsm(rho, bs_lc)
        lsm_gg = lc_to_gg(lsm_lc, Path(__file__).parent.parent  / "base_stations.csv")

    st.session_state["lsm_res"] = {
        "lc":  lsm_lc,
        "gg":  lsm_gg,
    }

    st.success("LSM finished – you can now run the Kalman filter.")

# ──────────────────────────────────────────────────────────────────────
#  2. Run / cache Kalman filter
# ──────────────────────────────────────────────────────────────────────
if kf_btn and "lsm_res" in st.session_state:
    with st.spinner("Kalman filtering …"):
        est_EN = kf_smooth(
            positions=st.session_state["lsm_res"]["lc"][:, :2],
            sigma_model_pos=q_pos,
            sigma_model_vel=q_vel,
            sigma_obs=r_meas,
        )
        # Combine filtered E,N with original U
        kf_lc = np.column_stack([
            est_EN,
            st.session_state["lsm_res"]["lc"][:, 2]
        ])
        kf_gg = lc_to_gg(kf_lc, Path(__file__).parent.parent / "base_stations.csv")

    st.session_state["kf_res"] = {
        "lc": kf_lc,
        "gg": kf_gg,
    }
    st.success("Kalman filtering done!")
    st.balloons()


# ──────────────────────────────────────────────────────────────────────
#  3. Map visualisation with layer checkboxes
# ──────────────────────────────────────────────────────────────────────
st.header("Trajectory comparison")

cols = st.columns(3)
show_true = cols[0].checkbox("Show kinematic data", value=True)
show_lsm  = cols[1].checkbox("Show Least-Squares", value="lsm_res" in st.session_state)
show_kf   = cols[2].checkbox("Show Kalman", value="kf_res" in st.session_state)

# default center
mean_lat, mean_lon = df_true.Latitude.mean(), df_true.Longitude.mean()
m = folium.Map(location=[mean_lat, mean_lon], zoom_start=18, max_zoom=30)

# true track
if show_true:
    folium.PolyLine(
        list(zip(df_true.Latitude, df_true.Longitude)),
        color="green", weight=4, tooltip="True trajectory"
    ).add_to(m)

# LSM
if show_lsm and "lsm_res" in st.session_state:
    latlon = st.session_state["lsm_res"]["gg"][:, :2]
    for i, (lat, lon) in enumerate(latlon):
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            color="blue",
            fill=True,
            fill_opacity=0.5,
            tooltip=f"LSM {i+1}",
        ).add_to(m)

# Kalman
if show_kf and "kf_res" in st.session_state:
    latlon = st.session_state["kf_res"]["gg"][:, :2]
    folium.PolyLine(
        latlon, color="red", weight=4, tooltip="Kalman trajectory"
    ).add_to(m)

folium.LayerControl().add_to(m)
st_folium(m, width=800, key="trajectory_map")
