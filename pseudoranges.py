import numpy as np
import pandas as pd

df_T1 = pd.read_csv('./KinematicData/T1.csv')
df_BS = pd.read_csv('./base_stations.csv')

T1 = df_T1[["Latitude", "Longitude", "Height"]].to_numpy()
BS = df_BS[["Latitude", "Longitude", "Height"]].to_numpy()

print("T1 shape: ", T1.shape)
print("BS shape: ", BS.shape)


from transformations import X_GG_GC, R_GC_LC

# transform to rad
BS_GG = [np.array([np.deg2rad(X[0]), np.deg2rad(X[1]), X[2]]) for X in BS]

# Base stations transformed to Global Cartesian coordinates
BS_GC = [X_GG_GC(X) for X in BS_GG]
O_GG = BS_GG[0]
O_GC = BS_GC[0]
# Base stations transformed to Local Cartesian coordinates

base_stations_LC = [R_GC_LC(O_GG) @ (X - O_GC) for X in BS_GC]



# Transform User Equipment (UE) coordinates to Local Cartesian coordinates
UE_GG = np.array([np.deg2rad(T1[:,0]), np.deg2rad(T1[:,1]), T1[:,2]]).T
UE_GC = [X_GG_GC(X) for X in UE_GG]
UE_LC = [R_GC_LC(O_GG) @ (X - O_GC) for X in UE_GC]
print("User Equipment in Local Cartesian coordinates:")
for i, ue in enumerate(UE_LC):
    print(f"UE{i+1}: {ue}")

# New height of the base stations
mean_ue_height = np.mean([u[2] for u in UE_LC])
height_base_stations = mean_ue_height + 20
base_stations_LC = np.array(base_stations_LC)
base_stations_LC[:, 2] = height_base_stations

# Calculate pseudoranges
rho_bs_ue = np.zeros((len(UE_LC), len(base_stations_LC)))
for i, ue in enumerate(UE_LC):
    for j, bs in enumerate(base_stations_LC):
        # Calculate the Euclidean distance between the base station and the user equipment
        distance = np.linalg.norm(bs - ue)
        rho_bs_ue[i, j] = distance


rho_bs_ue_noise = rho_bs_ue + np.random.normal(0, 1, size=rho_bs_ue.shape)

# Save the pseudoranges to a CSV file
pseudoranges_df = pd.DataFrame(rho_bs_ue_noise, columns=["BS1", "BS2", "BS3", "BS4", "BS5", "BS6", "BS7", "BS8"])
pseudoranges_df.to_csv('./pseudoranges.csv', index=False)