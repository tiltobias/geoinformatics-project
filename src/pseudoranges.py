"""
Pseudorange generator.

Given:
* a trajectory of user-equipment (UE) positions in geodetic (deg, deg, m),
* a list of base-station (BS) positions in geodetic (deg, deg, m),

the module

1. converts both to a local-Cartesian frame (origin = first BS),
2. sets all BS heights to (mean UE height + height_offset),   # like your notebook
3. computes true Euclidean ranges,
4. adds optional white Gaussian noise,
5. adds optional clock-offset bias (c · Δt).

Dependencies
------------
`transformations.py` must define

    X_GG_GC(φ, λ, h)      # geodetic → ECEF Cartesian
    R_GC_LC(origin_GG)    # 3×3 rotation ECEF → local-Cartesian (East-North-Up)

Units
-----
* Angles: **degrees** on input, **radians** internally.
* Heights / ranges / ENU coordinates: metres.
* Clock offset: **nanoseconds** (converted here to seconds).

"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from transformations import X_GG_GC, R_GC_LC

C_LIGHT = 299_792_458.0  # m s⁻¹


# ────────────────────────────────────────────────────────────────────────────
# 1. Basic helpers
# ────────────────────────────────────────────────────────────────────────────
def _load_xyz(csv_path: str | Path) -> np.ndarray:
    """
    Read *csv_path* and return an (N, 3) array with columns
    Latitude [deg], Longitude [deg], Height [m].

    Raises if those columns are missing.
    """
    df = pd.read_csv(csv_path)
    return df[["Latitude", "Longitude", "Height"]].to_numpy()


def _to_local_cartesian(
    gg_deg: np.ndarray,           # shape (N, 3)
    ref_gg_deg: np.ndarray,       # shape (3,) – origin (first BS)
) -> np.ndarray:
    """
    Convert geodetic (deg, deg, m) to local ENU (m) w.r.t. *ref_gg_deg*.
    """
    ref_gg_rad = np.deg2rad(ref_gg_deg)
    ref_gc     = X_GG_GC(ref_gg_rad)          # reference in ECEF
    R          = R_GC_LC(ref_gg_rad)          # 3×3 ECEF→LC rotation

    out = np.empty_like(gg_deg, dtype=float)
    for i, (φ_deg, λ_deg, h) in enumerate(gg_deg):
        gc  = X_GG_GC(np.deg2rad([φ_deg, λ_deg, h]))
        out[i] = R @ (gc - ref_gc)
    return out  # (N, 3)


# ────────────────────────────────────────────────────────────────────────────
# 2. Public API
# ────────────────────────────────────────────────────────────────────────────
def make(
    t1_csv: str | Path,
    bs_csv: str | Path,
    *,
    clock_offset_ns: np.ndarray | None = None,
    noise_sigma_m: float = 1.0,
    height_offset_m: float = 20.0,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pseudoranges ρ (shape [N_epochs, N_stations]).

    Returns
    -------
    ρ               : ndarray, true + noise + c·Δt
    UE_LC           : ndarray, UE positions in local ENU (m)
    BS_LC           : ndarray, BS positions in local ENU (m)

    Notes
    -----
    * If *clock_offset_ns* is supplied it must have length N_epochs;
      the same bias is applied to all base stations for the same epoch.
    * Set *noise_sigma_m = 0* for deterministic ranges.
    """
    # 2.1  Load geodetic coordinates (deg, deg, m)
    ue_gg = _load_xyz(t1_csv)          # (N, 3)
    bs_gg = _load_xyz(bs_csv)          # (M, 3)

    # 2.2  Convert to local Cartesian ENU
    ue_lc = _to_local_cartesian(ue_gg, ref_gg_deg=bs_gg[0])
    bs_lc = _to_local_cartesian(bs_gg, ref_gg_deg=bs_gg[0])

    # 2.3  Raise all BS to mean UE height + offset
    mean_h = ue_lc[:, 2].mean()
    bs_lc[:, 2] = mean_h + height_offset_m

    # 2.4  True ranges (Euclidean distance)
    rho_true = np.linalg.norm(
        ue_lc[:, None, :] - bs_lc[None, :, :], axis=-1
    )  # (N, M)

    # 2.5  Add white noise
    rng  = np.random.default_rng(seed)
    rho  = rho_true + rng.normal(0.0, noise_sigma_m, size=rho_true.shape)

    # 2.6  Add clock bias c · Δt
    if clock_offset_ns is not None:
        if len(clock_offset_ns) != len(ue_lc):
            raise ValueError("clock_offset_ns length must equal N_epochs")
        dt_sec = clock_offset_ns * 1e-9          # ns → s
        rho += C_LIGHT * dt_sec[:, None]         # broadcast to all BS

    return rho, ue_lc, bs_lc


def to_csv(
    rho: np.ndarray,
    path: str | Path = "pseudoranges.csv",
) -> Path:
    """
    Write pseudoranges to *path* with columns BS1, BS2, …
    """
    path = Path(path)
    n_bs = rho.shape[1]
    cols = [f"BS{i+1}" for i in range(n_bs)]
    pd.DataFrame(rho, columns=cols).to_csv(path, index=False)
    return path


# ────────────────────────────────────────────────────────────────────────────
# 3. Optional CLI for quick testing
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, sys

    p = argparse.ArgumentParser(description="Compute noisy pseudoranges.")
    p.add_argument("t1_csv", type=Path, help="UE trajectory CSV (Lat,Lon,H)")
    p.add_argument("bs_csv", type=Path, help="Base-station CSV (Lat,Lon,H)")
    p.add_argument("--clock", type=Path, help="Clock-offset CSV (one column, ns)")
    p.add_argument("--sigma", type=float, default=1.0, help="Noise σ [m]")
    p.add_argument("--seed",  type=int,   default=42,  help="RNG seed")
    p.add_argument("--out",   type=Path,  default=Path("pseudoranges.csv"))
    args = p.parse_args()

    clk = None
    if args.clock:
        clk = pd.read_csv(args.clock).to_numpy().squeeze()

    ρ, _, _ = make(
        args.t1_csv, args.bs_csv,
        clock_offset_ns=clk,
        noise_sigma_m=args.sigma,
        seed=args.seed,
    )
    to_csv(ρ, args.out)
    print(f"Wrote {args.out}", file=sys.stderr)
