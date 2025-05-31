"""
Constant-velocity Kalman smoother for 2-D trajectories
( East, North coordinates ).

State vector
------------
x = [E, N, Ẋ, Ṅ]ᵀ          [m, m, m/s, m/s]

Model (Δt = 1 epoch)
--------------------
xₖ =  F xₖ₋₁  +  w
yₖ =  H xₖ    +  v

    F = [[1 0 1 0],           process noise  ~  N(0, Q)
         [0 1 0 1],
         [0 0 1 0],
         [0 0 0 1]]

    H = [[1 0 0 0],           measurement noise ~ N(0, R)
         [0 1 0 0]]

Only *positions* are observed.

Public helpers
--------------
load_positions()      – read CSV with cols E, N
smooth()              – run the Kalman filter
plot()                – quick matplotlib preview
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ────────────────────────────────────────────────────────────────────────────
# 1.  I/O helper
# ────────────────────────────────────────────────────────────────────────────
def load_positions(csv_path: str | Path, cols=("E", "N")) -> np.ndarray:
    """Return Nx2 array [E, N] from *csv_path*."""
    return pd.read_csv(csv_path)[list(cols)].to_numpy()


# ────────────────────────────────────────────────────────────────────────────
# 2.  Kalman smoother
# ────────────────────────────────────────────────────────────────────────────
def smooth(
    positions: np.ndarray,
    *,
    sigma_model_pos: float = 1.0,      # Q position  σ  [m]
    sigma_model_vel: float = 1.0,      # Q velocity  σ  [m/s]
    sigma_obs: float = 5.0,            # R position  σ  [m]
    init_err_pos: float = 100.0,       # P₀ position σ  [m]
    init_err_vel: float = 1.0,         # P₀ velocity σ  [m/s]
) -> np.ndarray:
    """
    Apply a constant-velocity Kalman filter to *positions*.

    Parameters
    ----------
    positions
        Array shape (N, 2) with columns [E, N] in metres.
    sigma_model_pos, sigma_model_vel
        Std-dev of the process noise Q for positions and velocities.
    sigma_obs
        Std-dev of measurement noise R.
    init_err_pos, init_err_vel
        Initial covariance P₀ diagonal values (σ).

    Returns
    -------
    est_pos : ndarray
        Shape (N, 2) – filtered  (Ê, N̂)  positions.
    """
    # matrices
    F = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=float)

    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=float)

    Q = np.diag([sigma_model_pos**2,
                 sigma_model_pos**2,
                 sigma_model_vel**2,
                 sigma_model_vel**2])

    R = np.diag([sigma_obs**2, sigma_obs**2])

    I = np.eye(4)

    # initial state (use first observation, zero velocity)
    x_hat = np.array([[positions[0, 0]],
                      [positions[0, 1]],
                      [0.0],
                      [0.0]])

    P = np.diag([init_err_pos**2,
                 init_err_pos**2,
                 init_err_vel**2,
                 init_err_vel**2])

    # output array
    est = np.empty_like(positions, dtype=float)

    # Kalman loop
    for k, (E_meas, N_meas) in enumerate(positions):
        # 1. predict
        x_pred = F @ x_hat
        P_pred = F @ P @ F.T + Q

        # 2. update
        y = np.array([[E_meas], [N_meas]])            # innovation
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)           # Kalman gain

        x_hat = x_pred + K @ (y - H @ x_pred)
        P     = (I - K @ H) @ P_pred

        est[k] = x_hat[:2, 0]                         # save Ê, N̂

    return est


# ────────────────────────────────────────────────────────────────────────────
# 3.  Quick plot helper
# ────────────────────────────────────────────────────────────────────────────
def plot(raw: np.ndarray, est: np.ndarray) -> None:
    """Scatter *raw* positions and line plot of *est*."""
    plt.plot(raw[:, 0], raw[:, 1], "b.", markersize=3, label="Data")
    plt.plot(est[:, 0], est[:, 1], "r-", linewidth=2, label="Estimated")
    plt.plot(raw[0, 0], raw[0, 1], "go", label="Start")
    plt.legend()
    plt.axis("equal")
    plt.title("Kalman-smoothed trajectory")
    plt.xlabel("E [m]")
    plt.ylabel("N [m]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ────────────────────────────────────────────────────────────────────────────
# 4.  CLI entry point for ad-hoc runs
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, sys

    p = argparse.ArgumentParser(description="Smooth E,N positions with a CV Kalman filter.")
    p.add_argument("csv_in", type=Path, help="CSV with columns E,N")
    p.add_argument("--csv-out", type=Path, help="Write filtered positions to CSV")
    p.add_argument("--plot", action="store_true", help="Show matplotlib plot")
    p.add_argument("--sigma-model-pos", type=float, default=1.0)
    p.add_argument("--sigma-model-vel", type=float, default=1.0)
    p.add_argument("--sigma-obs",       type=float, default=5.0)
    args = p.parse_args()

    raw = load_positions(args.csv_in)
    est = smooth(
        raw,
        sigma_model_pos=args.sigma_model_pos,
        sigma_model_vel=args.sigma_model_vel,
        sigma_obs=args.sigma_obs,
    )

    if args.csv_out:
        pd.DataFrame(est, columns=["E_hat", "N_hat"]).to_csv(args.csv_out, index=False)
        print(f"Wrote {args.csv_out}", file=sys.stderr)

    if args.plot:
        plot(raw, est)
