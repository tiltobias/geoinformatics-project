from transformations import X_GG_GC, R_GC_LC, X_GC_GG
import numpy as np
import pandas as pd

# Load user position in Local Cartesian coordinates (skip 't' column)
user_position_LC = pd.read_csv('./user_position_LSM.csv')[["E", "N", "U"]].to_numpy()

# Load base station global geodetic coordinates (skip 'Title')
base_stations_df = pd.read_csv('./base_stations.csv')
base_stations_GG = base_stations_df[["Latitude", "Longitude", "Height"]].to_numpy()

# Step 1: Get the origin of the local coordinate system in GG (degrees) and convert to radians
X_O_GG = base_stations_GG[0]
lat = float(X_O_GG[0])
lon = float(X_O_GG[1])
h = float(X_O_GG[2])
X_O_GG_rad = np.array([
    np.deg2rad(lat),
    np.deg2rad(lon),
    h
])

# Step 2: Convert origin from GG to GC
X_O_GC = X_GG_GC(X_O_GG_rad)

# Step 3: Compute rotation matrix from GC to LC (based on origin GG coords)
R = R_GC_LC(X_O_GG_rad)

# Step 4: Transform user positions from LC to GC
user_position_GC = np.array([X_O_GC + R.T @ X for X in user_position_LC])

# Step 5: Convert each GC position to GG (in radians)
user_position_GG_rad = np.array([X_GC_GG(X) for X in user_position_GC])
user_position_GG_deg = np.rad2deg(user_position_GG_rad)

# Step 6: Fix height using origin height + local U
user_position_GG_deg[:, 2] = h + user_position_LC[:, 2]

# Step 7: Save to CSV
user_position_GG_df = pd.DataFrame(user_position_GG_deg, columns=["Latitude", "Longitude", "Height"])
user_position_GG_df.to_csv('./user_position_GG.csv', index=False)
