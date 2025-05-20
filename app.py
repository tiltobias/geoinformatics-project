# app.py
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from pathlib import Path

# ────────────────────────────────
# 1. Load data only once
# ────────────────────────────────
df = pd.read_csv(Path(__file__).parent / "KinematicData" / "T1.csv")

# ────────────────────────────────
# 2. Sidebar widgets (unchanged)
# ────────────────────────────────
st.sidebar.header("Simulation parameters")
sigma = st.sidebar.number_input("Noise σ (m)", 0.0, 10.0, 1.0)
step  = st.sidebar.number_input("Clock random-walk step (ns)", 0.0, 5.0, 0.5)
seed  = st.sidebar.number_input("Random seed", 0, 2**32 - 1, 42)

# ────────────────────────────────
# 3. Toggle logic
# ────────────────────────────────
if "show_markers" not in st.session_state:          # initialise the flag
    st.session_state.show_markers = False

if st.button("Show / hide true user-equipment positions"):
    # flip the flag every click
    st.session_state.show_markers = not st.session_state.show_markers

# ────────────────────────────────
# 4. Build the map *once* per run
# ────────────────────────────────

# Mean position of the user equipment
mean_lat = df.Latitude.mean()
mean_lon = df.Longitude.mean()

m = folium.Map(location=[mean_lat, mean_lon], zoom_start=18, max_zoom=30)

if st.session_state.show_markers:                  # add markers on demand
    for _, row in df.iterrows():
        folium.CircleMarker(
            [row.Latitude, row.Longitude],
            radius=5,
            color="green",
            fill=True, fill_opacity=0.6,
            popup="Measured",
        ).add_to(m)

st_folium(m, width=725, key="main_map")            # one call, stable key
