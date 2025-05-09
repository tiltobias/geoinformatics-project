import numpy as np
import pandas as pd

df = pd.read_csv('./KinematicData/T1.csv')
mean_ue_height = np.mean(df['Height'])
height_base_stations = mean_ue_height + 20

base_stations = [
    np.array([45.3435, 9.0102, height_base_stations]), # bottom left corner BS1
    np.array([45.3435, 9.0149, height_base_stations]), # bottom right corner BS2
    np.array([45.3451, 9.0149, height_base_stations]), # top right corner BS3
    np.array([45.3451, 9.0102, height_base_stations]), # top left corner BS4
    np.array([45.3438, 9.0109, height_base_stations]), # left roundabout BS5
    np.array([45.3446, 9.0137, height_base_stations]), # right roundabout BS6
    np.array([45.3446, 9.0126, height_base_stations]), # middle top BS7
    np.array([45.3436, 9.0125, height_base_stations]), # middle bottom BS8
]

from transformations import X_GG_GC, R_GC_LC

# transform to rad
base_stations_GG = [np.array([np.deg2rad(BS[0]), np.deg2rad(BS[1]), BS[2]]) for BS in base_stations]

# Base stations transformed to Global Cartesian coordinates
base_stations_GC = [X_GG_GC(BS) for BS in base_stations_GG]
origin_GG = base_stations_GG[0]
origin_GC = base_stations_GC[0]
# Base stations transformed to Local Cartesian coordinates

base_stations_LC = [R_GC_LC(origin_GG) @ (BS - origin_GC) for BS in base_stations_GC]

print("Base stations in Local Cartesian coordinates:")
for i, BS in enumerate(base_stations_LC):
    print(f"BS{i+1}: {BS}")
print("Base stations in Global Cartesian coordinates:")
for i, BS in enumerate(base_stations_GC):
    print(f"BS{i+1}: {BS}")
print("Base stations in Global Geodetic coordinates:")
for i, BS in enumerate(base_stations_GG):
    print(f"BS{i+1}: {BS}")

