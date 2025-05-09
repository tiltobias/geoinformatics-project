import numpy as np

def X_GG_GC(X_GG: np.ndarray) -> np.ndarray:
    # Transformation from ITRF Geodetic (GG) to Global Cartesian (GC) coordinates
    latitude = X_GG[0]
    longitude = X_GG[1]
    height = X_GG[2]
    a = 6378137.0
    b = 6356752.314140
    e_squared = (a**2 - b**2) / a**2
    R_N = a / np.sqrt(1 - e_squared * np.sin(latitude)**2)
    x_GC = (R_N + height) * np.cos(latitude) * np.cos(longitude)
    y_GC = (R_N + height) * np.cos(latitude) * np.sin(longitude)
    z_GC = (R_N * (1 - e_squared) + height) * np.sin(latitude)
    return np.array([x_GC, y_GC, z_GC])


def R_GC_LC(X_O_GG: np.ndarray) -> np.ndarray:
    # Rotation matrix from Global Cartesian (GC) to Local Cartesian (LC)
    R_0 = np.array([
        [-np.sin(X_O_GG[1]), np.cos(X_O_GG[1]), 0],
        [-np.sin(X_O_GG[0]) * np.cos(X_O_GG[1]), -np.sin(X_O_GG[0]) * np.sin(X_O_GG[1]), np.cos(X_O_GG[0])],
        [np.cos(X_O_GG[0]) * np.cos(X_O_GG[1]), np.cos(X_O_GG[0]) * np.sin(X_O_GG[1]), np.sin(X_O_GG[0])]
    ])
    return R_0
