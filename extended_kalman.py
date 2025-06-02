import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# === Load data ===
pseudoranges = pd.read_csv("pseudoranges.csv").to_numpy()
base_stations = pd.read_csv("base_stations_LC.csv").to_numpy()

# Constants
c = 299_792_458  # Speed of light in m/s
dt = 1.0         # Time step in seconds
U_u = 20.0       # Fixed height of the receiver

num_steps, num_stations = pseudoranges.shape

# === Initial State Estimate ===
initial_pos = np.mean(base_stations[:, :2], axis=0)  # Mean of (E, N)
x = np.array([initial_pos[1], initial_pos[0], 0.0, 0.0, 0.0])  # [N, E, v_N, v_E, dt]

# Initial covariance
P_cov = np.eye(5) * 100.0

# === Motion Model ===
F = np.eye(5)
F[0, 2] = dt  # N update with v_N
F[1, 3] = dt  # E update with v_E

# Process and measurement noise
Q = np.diag([1.0, 1.0, 0.5, 0.5, (0.5e-9 * c)**2])  # Process noise
R = np.eye(num_stations) * 25.0  # Measurement noise

estimated_positions = []

# === Extended Kalman Filter Loop ===
for k in range(num_steps):
    # --- Prediction ---
    x = F @ x
    P_cov = F @ P_cov @ F.T + Q

    # --- Measurement Model ---
    z = pseudoranges[k]
    h = []
    H = []

    for i in range(num_stations):
        E_i, N_i, U_i = base_stations[i]
        dN = N_i - x[0]
        dE = E_i - x[1]
        dU = U_i - U_u

        rho = np.sqrt(dN**2 + dE**2 + dU**2)
        if rho < 1e-6 or np.isnan(rho):
            continue

        h.append(rho + c * x[4])  # h(x)
        
        Hi = np.zeros(5)
        Hi[0] = -dN / rho
        Hi[1] = -dE / rho
        Hi[4] = c
        H.append(Hi)

    if len(h) < 4:
        # Not enough valid measurements
        estimated_positions.append([x[0], x[1]])
        continue

    h = np.array(h)
    H = np.vstack(H)
    R_used = np.eye(len(h)) * 25.0

    z_used = z[:len(h)]  # Use matching subset
    y = z_used - h       # Innovation

    # --- Kalman Gain ---
    S = H @ P_cov @ H.T + R_used
    try:
        K = P_cov @ H.T @ np.linalg.inv(S)
    except np.linalg.LinAlgError:
        print(f"[Warning] SVD did not converge at step {k}, using pseudo-inverse.")
        K = P_cov @ H.T @ np.linalg.pinv(S + 1e-6 * np.eye(S.shape[0]))

    # --- Update ---
    x = x + K @ y
    P_cov = (np.eye(5) - K @ H) @ P_cov

    estimated_positions.append([x[0], x[1]])

# === Plotting ===
estimated_positions = np.array(estimated_positions)

plt.figure(figsize=(10, 6))
plt.plot(estimated_positions[:, 1], estimated_positions[:, 0], label='Estimated Path', color='blue')
plt.scatter(base_stations[:, 0], base_stations[:, 1], c='red', marker='x', s=100, label='Base Stations')
plt.xlabel("East [m]")
plt.ylabel("North [m]")
plt.title("Estimated 2D Path with EKF")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()
