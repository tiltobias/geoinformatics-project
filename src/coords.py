"""
Coordinate helpers.

lc_to_gg() converts a trajectory from local-Cartesian (ENU, metres) back to
geodetic latitude / longitude / height, using the first base-station as the
local origin – exactly the same logic you prototyped in your script.

The function is deliberately lightweight: no plotting, no Streamlit calls, no
file I/O unless you ask for it.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from transformations import X_GG_GC, R_GC_LC, X_GC_GG


# ────────────────────────────────────────────────────────────────────────────
# Main helper
# ────────────────────────────────────────────────────────────────────────────
def lc_to_gg(
    user_lc: np.ndarray,                # shape (N, 3)  [E, N, U]  in metres
    bs_csv: str | Path = "base_stations.csv",
    origin_index: int = 0,
) -> np.ndarray:
    """
    Convert local-Cartesian ENU positions to geodetic (deg, deg, m).

    Parameters
    ----------
    user_lc
        Trajectory in the local frame (East, North, Up).
    bs_csv
        CSV with columns Latitude, Longitude, Height for the base stations.
        The row *origin_index* defines the local origin.
    origin_index
        Which base-station row to use as the origin (default = 0).

    Returns
    -------
    user_gg_deg : ndarray, shape (N, 3)
        Columns: [Latitude°, Longitude°, Height m].
    """
    # 1. load origin base-station in geodetic (deg)
    bs_df = pd.read_csv(bs_csv)
    ref = bs_df.loc[origin_index, ["Latitude", "Longitude", "Height"]]
    ref = np.array([float(ref["Latitude"]), float(ref["Longitude"]), float(ref["Height"])])
    ref_rad = np.array([np.deg2rad(ref[0]), np.deg2rad(ref[1]), ref[2]])  # (φ_rad, λ_rad, h)


    # 2. build rotation + ECEF origin
    ref_gc = X_GG_GC(ref_rad)
    R = R_GC_LC(ref_rad)               # ECEF ➜ local-Cartesian

    # 3. LC ➜ ECEF ➜ GG
    user_gc = ref_gc + (R.T @ user_lc.T).T          # (N, 3)
    user_gg_rad = np.vstack([X_GC_GG(p) for p in user_gc])
    user_gg_deg = np.rad2deg(user_gg_rad)

    # 4. copy local Up as relative height
    user_gg_deg[:, 2] = ref[2] + user_lc[:, 2]

    return user_gg_deg


# ────────────────────────────────────────────────────────────────────────────
# Optional convenience wrapper
# ────────────────────────────────────────────────────────────────────────────
def to_csv(
    user_gg_deg: np.ndarray,
    path: str | Path = "user_position_GG.csv",
) -> Path:
    """Write the (N, 3) geodetic array to *path*."""
    path = Path(path)
    pd.DataFrame(user_gg_deg, columns=["Latitude", "Longitude", "Height"]).to_csv(
        path, index=False
    )
    return path


# ────────────────────────────────────────────────────────────────────────────
# CLI (for quick standalone testing)
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, sys

    p = argparse.ArgumentParser(description="Convert LC trajectory to lat/lon.")
    p.add_argument("user_lc_csv", type=Path, help="CSV with E,N,U columns")
    p.add_argument("bs_csv",      type=Path, help="Base-stations CSV")
    p.add_argument("--out",       type=Path, default=Path("user_position_GG.csv"))
    args = p.parse_args()

    user_lc = pd.read_csv(args.user_lc_csv)[["E", "N", "U"]].to_numpy()
    user_gg = lc_to_gg(user_lc, args.bs_csv)
    to_csv(user_gg, args.out)
    print(f"Wrote {args.out}", file=sys.stderr)
