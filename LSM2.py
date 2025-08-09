import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Inputs
P = pd.read_csv("pseudoranges.csv").to_numpy()                  
base_stations = pd.read_csv("base_stations_LC.csv").to_numpy()  

c = 299_792_458.0  # m/s

def LSM(P: np.ndarray, x0_EN: np.ndarray, base_ENU: np.ndarray,
        max_iter: int = 50, tol: float = 1e-6) -> np.ndarray:
    """
    2D iterative least squares with clock bias in seconds.
    State per epoch: [E, N, dt]
    Height is fixed to 0 in the local frame.
    Stopping rule: ||[dE, dN]||_2 < tol (ignore ddt).
    """
    n_epochs, m = P.shape
    X = np.zeros((n_epochs, 3))
    X[0, :] = np.r_[x0_EN[:2], 0.0]  # initial [E, N, dt]

    base_EN = base_ENU[:, :2]  # only E,N used
    for t in range(n_epochs):
        E, N, dt = X[t, :].copy()
        for i in range(max_iter):
            # geometric ranges in 2D (height fixed to 0)
            dEN = np.c_[E - base_EN[:, 0], N - base_EN[:, 1]]
            rho = np.linalg.norm(dEN, axis=1)  # (m,)

            # residuals: l = P - (rho + c*dt)
            l = P[t, :m] - (rho + c * dt)

            # guard against division by zero
            rho_safe = np.where(rho == 0.0, 1.0, rho)

            # design matrix A = [drho/dE, drho/dN, c]
            A = np.empty((m, 3))
            A[:, 0] = dEN[:, 0] / rho_safe
            A[:, 1] = dEN[:, 1] / rho_safe
            A[:, 2] = c

            # LS increment: dx = [dE, dN, ddt]
            dx, *_ = np.linalg.lstsq(A, l, rcond=None)

            # update
            E_new, N_new, dt_new = E + dx[0], N + dx[1], dt + dx[2]

            if np.max(np.abs(dx[:2])) < tol:   # only E and N, ∞-norm
                E, N, dt = E_new, N_new, dt_new
                print(f"Break at cycle: {i+1}  epoch: {t+1}  position: [{E:.3f}, {N:.3f}]  dt: {dt:.6e}s")
                break

            E, N, dt = E_new, N_new, dt_new

        X[t, :] = [E, N, dt]
        if t + 1 < n_epochs:
            X[t + 1, :] = X[t, :]  # warm start next epoch

    return X  


x0_EN = base_stations.mean(axis=0)[:2]  # centroid in EN
X_hat = LSM(P, x0_EN, base_stations, max_iter=50, tol=1e-6)


# With height fixed to 0
X_hat_height = np.c_[X_hat[:, 0], X_hat[:, 1], np.zeros(len(X_hat)), X_hat[:, 2]]
pd.DataFrame(X_hat_height, columns=["E", "N", "U", "t"]).to_csv("./user_position_LSM2.csv", index=False)

# Plot clock offset in seconds (use c*dt for meters if you prefer)
plt.figure(figsize=(10, 4))
plt.plot(X_hat[:, 2], label="Clock offset dt (s)")
plt.title("Clock Offset Over Time")
plt.xlabel("Epoch")
plt.ylabel("Clock offset (s)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
