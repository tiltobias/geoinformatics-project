import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
base_stations = pd.read_csv("base_stations_LC.csv").to_numpy()  # Columns: E, N, U
pseudoranges = pd.read_csv("pseudoranges.csv").to_numpy()        # Each row: pseudoranges from all stations at a time step

# Constants
c = 299_792_458  # Speed of light [m/s]
dt = 1.0         # Time step [s]
U_u = 20.0       # Fixed height of receiver [m] (assumed)

num_steps, num_stations = pseudoranges.shape

# Initial state: [N, E, v_N, v_E, clock_bias]
mean_E = np.mean(base_stations[:, 0])
mean_N = np.mean(base_stations[:, 1])
x = np.array([mean_N, mean_E, 0.0, 0.0, 0.0])  # [N, E, v_N, v_E, b]

# Initial covariance
P_cov = np.eye(5) * 100.0

# State transition model
F = np.eye(5)
F[0, 2] = dt  # N position update from velocity
F[1, 3] = dt  # E position update from velocity

# Process noise
Q = np.diag([1.0, 1.0, 0.5, 0.5, (0.5e-9 * c) ** 2])

# Measurement noise
R_scalar = 25.0

# Store estimated positions
estimated_positions = []

for k in range(num_steps):
    # Predict
    x = F @ x
    P_cov = F @ P_cov @ F.T + Q

    # Measurement model
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

        h.append(rho + c * x[4])  # predicted pseudorange
        H_i = np.zeros(5)
        H_i[0] = -dN / rho
        H_i[1] = -dE / rho
        H_i[4] = c
        H.append(H_i)

    if len(h) < 4:
        estimated_positions.append([x[0], x[1]])
        continue

    h = np.array(h)
    H = np.vstack(H)
    z_used = z[:len(h)]
    y = z_used - h
    R = np.eye(len(h)) * R_scalar

    S = H @ P_cov @ H.T + R
    try:
        K = P_cov @ H.T @ np.linalg.inv(S)
    except np.linalg.LinAlgError:
        K = P_cov @ H.T @ np.linalg.pinv(S + 1e-6 * np.eye(S.shape[0]))

    x = x + K @ y
    P_cov = (np.eye(5) - K @ H) @ P_cov

    estimated_positions.append([x[0], x[1]])

estimated_positions = np.array(estimated_positions)

# Plot the results
plt.figure(figsize=(10, 6))
# We skip the first three points to avoid initial noise
plt.plot(estimated_positions[3:, 1], estimated_positions[3:, 0], label="Estimated Path", color='blue')
plt.scatter(base_stations[:, 0], base_stations[:, 1], c='red', marker='x', s=100, label="Base Stations")
plt.xlabel("East [m]")
plt.ylabel("North [m]")
plt.title("Estimated 2D Path Using Extended Kalman Filter")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()
