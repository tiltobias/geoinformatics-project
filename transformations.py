import numpy as np
"""
These functions are from the solution of Positioning&LBS exercise 1.
Author: Tobias Andresen
"""

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

def X_GC_GG(X_GC: np.ndarray) -> np.ndarray:
    # Transformation from Global Cartesian (GC) to ITRF Geodetic (GG)
    a = 6378137.0
    b = 6356752.314140
    e_squared = (a**2 - b**2) / a**2
    e_b_squared = (a**2 - b**2) / b**2
    r = np.sqrt(X_GC[0]**2 + X_GC[1]**2)
    psi = np.arctan2(X_GC[2] , (r * np.sqrt(1 - e_squared)))
    lambda_ = np.arctan2(X_GC[1] , X_GC[0])
    phi = np.arctan2((X_GC[2] + e_b_squared * b * np.sin(psi)**3) , (r - e_squared * a * np.cos(psi)**3))
    R_N = a / np.sqrt(1 - e_squared * np.sin(phi)**2)
    h = r / np.cos(phi) - R_N
    return np.array([phi, lambda_, h])


def R_GC_LC(X_O_GG: np.ndarray) -> np.ndarray:
    # Rotation matrix from Global Cartesian (GC) to Local Cartesian (LC)
    R_0 = np.array([
        [-np.sin(X_O_GG[1]), np.cos(X_O_GG[1]), 0],
        [-np.sin(X_O_GG[0]) * np.cos(X_O_GG[1]), -np.sin(X_O_GG[0]) * np.sin(X_O_GG[1]), np.cos(X_O_GG[0])],
        [np.cos(X_O_GG[0]) * np.cos(X_O_GG[1]), np.cos(X_O_GG[0]) * np.sin(X_O_GG[1]), np.sin(X_O_GG[0])]
    ])
    return R_0
