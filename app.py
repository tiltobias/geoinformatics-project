# app.py
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

st.sidebar.header("Simulation parameters")
sigma = st.sidebar.number_input("Noise σ (m)", 0.0, 10.0, 1.0)
step  = st.sidebar.number_input("Clock random-walk step (ns)", 0.0, 5.0, 0.5)
seed  = st.sidebar.number_input("Random-seed", 0, 2**32-1, 42)

# Add map 
m = folium.Map(location= [45.3435, 9.0102], zoom_start=16)
# Render map in streamlit
st_data = st_folium(m, width=725)

# if st.sidebar.button("Run simulation"):
#     pr_df, clk_df = run_simulation(sigma, step, seed)
#     # --- Estimators
#     lsm_pos  = lsm(pr_df, clk_df)
#     kf_pos   = kf(lsm_pos)
#     ekf_pos  = ekf(pr_df, clk_df)
#     # --- Downloads
#     for name, df in [("LSM", lsm_pos), ("KF", kf_pos), ("EKF", ekf_pos)]:
#         csv = df.to_csv(index=False).encode()
#         st.download_button(f"Download {name} CSV", csv, file_name=f"{name}.csv")
