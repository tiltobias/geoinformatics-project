# src/positioning.py
"""
Least–squares multilateration (epoch-by-epoch).

The core logic is exactly the same as your notebook snippet; nothing has been
changed other than:

* moving the code into a function (`lsm`)
* making the initial guess `x0` optional (defaults to the mean of the
  base-station coordinates)
* returning the (N, 3) ENU positions so the Streamlit app can feed them
  directly to `coords.lc_to_gg`.
"""

from __future__ import annotations
import numpy as np

C_LIGHT = 299_792_458.0  # speed of light [m s⁻¹]


def lsm(
    P: np.ndarray,                 # pseudoranges  [N_epochs, N_stations]
    base_stations: np.ndarray,     # ENU coords    [N_stations, 3]
    x0: np.ndarray | None = None,  # initial guess [3,]  (E,N,U)
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
    verbose: bool = False,
) -> np.ndarray:
    """
    Epoch-wise least-squares position fix with clock-bias estimation.

    Returns
    -------
    user_pos_lc : ndarray, shape (N_epochs, 3)
        Estimated user-equipment positions in the *same local frame* as the
        base stations (columns: E, N, U).
    """
    if x0 is None:
        # crude but common: start at the mean of the station locations
        x0 = base_stations.mean(axis=0)

    n_epochs, n_stations = P.shape
    user_position = np.zeros((n_epochs, 4))          # E, N, U, clockBias
    user_position[0, :3] = x0                        # first guess
    # clock bias initialised to 0 by default

    # --- helper functions ----------------------------------------------------
    def jacobian(x: np.ndarray) -> np.ndarray:
        """Build A(x) Jacobian for the current state."""
        A = np.zeros((n_stations, 4))
        for i in range(n_stations):
            dist = np.linalg.norm(x[:3] - base_stations[i])
            A[i, :3] = (x[:3] - base_stations[i]) / dist
            A[i, 3] = 1.0
        return A

    def delta_p(x: np.ndarray, epoch: int) -> np.ndarray:
        """Observation minus computed pseudoranges at *epoch*."""
        dP = np.zeros(n_stations)
        for i in range(n_stations):
            dist = np.linalg.norm(x[:3] - base_stations[i])
            dP[i] = P[epoch, i] - dist
        return dP
    # -------------------------------------------------------------------------

    for epoch in range(n_epochs):
        x = user_position[epoch].copy()              # start from last epoch

        for it in range(max_iter):
            A_  = jacobian(x)
            dPr = delta_p(x, epoch)
            dx, *_ = np.linalg.lstsq(A_, dPr, rcond=None)

            # update position & clock bias (convert bias metres → seconds)
            x[:3] += dx[:3]
            x[3]  += dx[3] / C_LIGHT

            if np.linalg.norm(dx) < tol:
                if verbose:
                    print(f"Converged in {it+1:2d} iters @ epoch {epoch+1:4d}")
                break

        # store result and propagate to next epoch
        user_position[epoch] = x
        if epoch + 1 < n_epochs:
            user_position[epoch + 1] = x

    return user_position[:, :3]      # return only E, N, U


# optional: quick CLI for ad-hoc checks ------------------------------
if __name__ == "__main__":
    import argparse, sys, pandas as pd, pathlib as pl

    p = argparse.ArgumentParser(description="Run LSM on pseudoranges CSVs.")
    p.add_argument("pseudoranges_csv", type=pl.Path)
    p.add_argument("base_stations_csv", type=pl.Path)
    p.add_argument("--out", type=pl.Path, default=pl.Path("user_position_LSM.csv"))
    args = p.parse_args()

    P   = pd.read_csv(args.pseudoranges_csv).to_numpy()
    BS  = pd.read_csv(args.base_stations_csv).to_numpy()

    est = lsm(P, BS)
    pd.DataFrame(est, columns=["E", "N", "U"]).to_csv(args.out, index=False)
    print(f"Wrote {args.out}", file=sys.stderr)
