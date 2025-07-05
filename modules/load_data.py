import numpy as np
from transformations import X_GC_GG, X_GG_GC, R_GC_LC

def O_GG_rad(O_GG):
    """
    Convert origin in Global Geodetic coordinates to radians.
    Args:
        O_GG (np.ndarray): Origin in Global Geodetic coordinates (latitude, longitude, height).
    Returns:
        np.ndarray: Origin in radians (latitude, longitude, height).
    """
    return np.array([np.deg2rad(O_GG[0]), np.deg2rad(O_GG[1]), O_GG[2]])

def create_origin(base_stations, base_station_height=20.):
    """
    Create the origin in Global Cartesian coordinates from base stations.
    Args:
        base_stations (np.ndarray): Array of base stations in Global Geodetic coordinates (latitude, longitude, height).
        base_station_height (float): Height of the base stations above ground level.
    Returns:
        tuple: Origin in Global Geodetic coordinates (O_GG) and in Global Cartesian coordinates (O_GC).
    """
    O_GG = np.mean(base_stations, axis=0)
    O_GG[2] = O_GG[2] - base_station_height # Adjust height to ground level

    O_GC = X_GG_GC(O_GG_rad(O_GG))
    return O_GG, O_GC

def transform_GG_to_LC(geodetic_coords, origin_GG):
    """
    Transform coordinates from Global Geodetic to Local Cartesian coordinates.
    Args:
        geodetic_coords (np.ndarray): Array of coordinates in Global Geodetic coordinates (latitude, longitude, height).
        origin_GG (np.ndarray): Origin in Global Geodetic coordinates.
    Returns:
        np.ndarray: Coordinates in Local Cartesian coordinates.
    """
    O_GC = X_GG_GC(O_GG_rad(origin_GG))
    return np.array([R_GC_LC(O_GG_rad(origin_GG)) @ (X - O_GC) for X in geodetic_coords])

def transform_LC_to_GG(local_coords, origin_GG):
    """
    Transform coordinates from Local Cartesian to Global Geodetic coordinates.
    Args:
        local_coords (np.ndarray): Array of coordinates in Local Cartesian coordinates.
        origin_GG (np.ndarray): Origin in Global Geodetic coordinates.
    Returns:
        np.ndarray: Coordinates in Global Geodetic coordinates.
    """
    O_GC = X_GG_GC(O_GG_rad(origin_GG))
    return np.array([X_GC_GG(O_GC + R_GC_LC(O_GG_rad(origin_GG)).T @ X) for X in local_coords])


