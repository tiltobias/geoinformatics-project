from transformations import X_GG_GC, R_GC_LC, X_GC_GG
import numpy as np
import pandas as pd

user_position_LC = pd.read_csv('./user_position_LC.csv').to_numpy()
base_stations_LC = pd.read_csv('./base_stations_LC.csv').to_numpy()
base_stations_GC = pd.read_csv('./base_stations.csv').to_numpy()

# Transform user position from Local Cartesian coordinates to Global Cartesian coordinates
# Origin is the first base station
O_GC = base_stations_LC[0]
user_position_GC = np.array([R_GC_LC(O_GC) @ (X - O_GC) for X in user_position_LC])
user_position_GG = np.array([X_GC_GG(X) for X in user_position_GC])
user_position_GG = np.rad2deg(user_position_GG)
user_position_GG_df = pd.DataFrame(user_position_GG, columns=["Latitude", "Longitude", "Height"])
user_position_GG_df.to_csv('./user_position_GG.csv', index=False)


X_O_GG = base_stations_GC[0]
X_O_GG_rad = np.array([
    np.deg2rad(X_O_GG[0]), 
    np.deg2rad(X_O_GG[1]), 
    float(X_O_GG[2])
])
X_O_GC = X_GG_GC(X_O_GG_rad)
from transformations import R_GC_LC # Task 4
print("\nTask 4")
R_GC_LC_ = R_GC_LC(X_O_GG_rad)
user_position_GC = X_O_GC + R_GC_LC_.T @ user_position_LC
