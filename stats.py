import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ex_kalman = pd.read_csv("ex_kalman_LC.csv").to_numpy()
kalman = pd.read_csv("kalman_LC.csv").to_numpy()
lsm = pd.read_csv("user_position_LSM.csv").to_numpy()
base_stations = pd.read_csv("base_stations_LC.csv").to_numpy()
pseudoranges  = pd.read_csv("pseudoranges.csv").to_numpy()
T1 = pd.read_csv("UE_LC.csv").to_numpy()

#Convert T1 to Local Cartesian coordinates

c = 299792458  # Speed of light in m/s

# --- LEAST SQUARES METHOD ---

mean_east_error_LSM = np.mean(lsm[:, 0] - T1[:, 0])
print(f"Mean East Error LSM: {mean_east_error_LSM:.3f} m")
meadian_east_error_LSM = np.median(lsm[:, 0] - T1[:, 0])
print(f"Median East Error LSM: {meadian_east_error_LSM} m")
mean_north_error_LSM = np.mean(lsm[:, 1] - T1[:, 1])
print(f"Mean North Error LSM: {mean_north_error_LSM:.3f} m")
meadian_north_error_LSM = np.median(lsm[:, 1] - T1[:, 1])
print(f"Median North Error LSM: {meadian_north_error_LSM:.3f} m")

#2D statistics
err_LSM = np.linalg.norm(lsm[:, :2]   - T1[:, :2], axis=1)
print(f"Mean 2D Error LSM: {np.mean(err_LSM):.3f} m")
print(f"Median 2D Error LSM: {np.median(err_LSM):.3f} m")
print(f"Standard Deviation 2D Error LSM: {np.std(err_LSM):.3f} m")
print(f"Maximum 2D Error LSM: {np.max(err_LSM):.3f} m")

# Standard deviation of East and North errors
std_east_error_LSM = np.std(lsm[:, 0] - T1[:, 0])
print(f"Standard Deviation East Error LSM: {std_east_error_LSM:.3f} m")
std_north_error_LSM = np.std(lsm[:, 1] - T1[:, 1])
print(f"Standard Deviation North Error LSM: {std_north_error_LSM:.3f} m")

# Maximum in modulus of East and North errors
max_east_error_LSM = np.max(np.abs(lsm[:, 0] - T1[:, 0]))
print(f"Maximum East Error LSM: {max_east_error_LSM:.3f} m")
max_north_error_LSM = np.max(np.abs(lsm[:, 1] - T1[:, 1]))
print(f"Maximum North Error LSM: {max_north_error_LSM:.3f} m")

# --- KALMAN FILTER ---
mean_east_error_KF = np.mean(kalman[:, 0] - T1[:, 0])
print(f"Mean East Error KF: {mean_east_error_KF:.3f} m")
meadian_east_error_KF = np.median(kalman[:, 0] - T1[:, 0])
print(f"Median East Error KF: {meadian_east_error_KF:.3f} m")
mean_north_error_KF = np.mean(kalman[:, 1] - T1[:, 1])
print(f"Mean North Error KF: {mean_north_error_KF:.3f} m")
meadian_north_error_KF = np.median(kalman[:, 1] - T1[:, 1])
print(f"Median North Error KF: {meadian_north_error_KF:.3f} m") 
# Standard deviation of East and North errors
std_east_error_KF = np.std(kalman[:, 0] - T1[:, 0])
print(f"Standard Deviation East Error KF: {std_east_error_KF:.3f} m")
std_north_error_KF = np.std(kalman[:, 1] - T1[:, 1])
print(f"Standard Deviation North Error KF: {std_north_error_KF:.3f} m")
# Maximum in modulus of East and North errors
max_east_error_KF = np.max(np.abs(kalman[:, 0] - T1[:, 0]))
print(f"Maximum East Error KF: {max_east_error_KF:.3f} m")
max_north_error_KF = np.max(np.abs(kalman[:, 1] - T1[:, 1]))
print(f"Maximum North Error KF: {max_north_error_KF:.3f} m")     

# 2D statistics
err_KF = np.linalg.norm(kalman[:, :2] - T1[:, :2], axis=1)
print(f"Mean 2D Error KF: {np.mean(err_KF):.3f} m")
print(f"Median 2D Error KF: {np.median(err_KF):.3f} m")
print(f"Standard Deviation 2D Error KF: {np.std(err_KF):.3f} m")
print(f"Maximum 2D Error KF: {np.max(err_KF):.3f} m")

# --- EXTENDED KALMAN FILTER ---
mean_east_error_EKF = np.mean(ex_kalman[:, 0] - T1[:, 0])
print(f"Mean East Error EKF: {mean_east_error_EKF:.3f} m")
meadian_east_error_EKF = np.median(ex_kalman[:, 0] - T1[:, 0])
print(f"Median East Error EKF: {meadian_east_error_EKF:.3f} m")
mean_north_error_EKF = np.mean(ex_kalman[:, 1] - T1[:, 1])  
print(f"Mean North Error EKF: {mean_north_error_EKF:.3f} m")
meadian_north_error_EKF = np.median(ex_kalman[:, 1] - T1[:, 1])
print(f"Median North Error EKF: {meadian_north_error_EKF:.3f} m")
# Standard deviation of East and North errors
std_east_error_EKF = np.std(ex_kalman[:, 0] - T1[:, 0])
print(f"Standard Deviation East Error EKF: {std_east_error_EKF:.3f} m")
std_north_error_EKF = np.std(ex_kalman[:, 1] - T1[:, 1])
print(f"Standard Deviation North Error EKF: {std_north_error_EKF:.3f} m")
# Maximum in modulus of East and North errors
max_east_error_EKF = np.max(np.abs(ex_kalman[:, 0] - T1[:, 0]))
print(f"Maximum East Error EKF: {max_east_error_EKF:.3f} m")
max_north_error_EKF = np.max(np.abs(ex_kalman[:, 1] - T1[:, 1]))
print(f"Maximum North Error EKF: {max_north_error_EKF:.3f} m")  

# 2D statistics
err_EKF = np.linalg.norm(ex_kalman[:, :2] - T1[:, :2], axis=1)
print(f"Mean 2D Error EKF: {np.mean(err_EKF):.3f} m")
print(f"Median 2D Error EKF: {np.median(err_EKF):.3f} m")
print(f"Standard Deviation 2D Error EKF: {np.std(err_EKF):.3f} m")
print(f"Maximum 2D Error EKF: {np.max(err_EKF):.3f} m")