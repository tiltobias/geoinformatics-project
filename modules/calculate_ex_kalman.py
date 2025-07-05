import numpy as np

"""
This script implements an Extended Kalman Filter (EKF) to estimate the user's position 
based on pseudorange data from 8 base stations.
Following professor's notes for EKF implementation.
"""

def calculate_ex_kalman(P, base_stations, dt=1.0, sigma_dt=0.5e-9, sigma_pos=1.0, sigma_vel=0.1, sigma_pseudorange=0.1):
    """
    Calculate the Extended Kalman Filter for the given pseudorange data.
    
    Args:
        P (np.ndarray): Measured pseudoranges.
        base_stations (np.ndarray): Array of base station positions in Local Cartesian coordinates.
        dt (float): Time step in seconds.
        sigma_dt (float): Clock drift standard deviation in seconds.
        sigma_pos (float): Position process noise standard deviation in meters.
        sigma_vel (float): Velocity process noise standard deviation in m/s.
        sigma_pseudorange (float): Pseudorange measurement noise standard deviation in meters.
        
    Returns:
        np.ndarray: Estimated user position and velocity over time. Local Cartesian coordinates (E, N, V_N, V_E, c*dt_u).
    """
    # Number of base stations and measurements
    n_stations = base_stations.shape[0]
    n_measurements = P.shape[0]

    print(f"Number of base stations: {n_stations}")
    print(f"Number of measurements: {n_measurements}")
    print(f"Base stations shape: {base_stations.shape}")
    print(f"Pseudoranges shape: {P.shape}")

    # State vector: [N_u, E_u, V_N, V_E, c*dt_u]
    # N_u, E_u: North and East positions
    # V_N, V_E: North and East velocities  
    # c*dt_u: Clock offset (in meters)

    # Initialize state vector
    # Starting guess: center of base stations
    initial_N = np.mean(base_stations[:, 1])  # North coordinate
    initial_E = np.mean(base_stations[:, 0])  # East coordinate
    initial_VN = 0.0  # Initial north velocity
    initial_VE = 0.0  # Initial east velocity
    initial_clock_offset = 0.0  # Initial clock offset

    x_hat = np.array([initial_N, initial_E, initial_VN, initial_VE, initial_clock_offset])
    print(f"Initial state estimate: {x_hat}")

    # State transition matrix T_k (constant velocity model)
    T = np.array([
        [1, 0, dt, 0, 0],  # N_u(k) = N_u(k-1) + dt * V_N(k-1)
        [0, 1, 0, dt, 0],  # E_u(k) = E_u(k-1) + dt * V_E(k-1)
        [0, 0, 1, 0, 0],   # V_N(k) = V_N(k-1)
        [0, 0, 0, 1, 0],   # V_E(k) = V_E(k-1)
        [0, 0, 0, 0, 1]    # c*dt_u(k) = c*dt_u(k-1)
    ])

    c = 299792458  # Speed of light in m/s
    # Process noise covariance matrix C_k^epsilon
    C_epsilon = np.diag([
        sigma_pos**2,  # Position N noise
        sigma_pos**2,  # Position E noise
        sigma_vel**2,  # Velocity N noise
        sigma_vel**2,  # Velocity E noise
        (c * sigma_dt)**2  # Clock offset noise
    ])

    # Initial error covariance matrix
    C_error = np.eye(5) * 100.0  # Large initial uncertainty (#TODO Assumption: 100 m uncertainty)

    # Measurement noise covariance matrix C_k^nu
    C_nu = np.eye(n_stations) * sigma_pseudorange**2

    def compute_predicted_pseudoranges(state, base_stations):
        """
        Compute predicted pseudoranges based on current state estimate.
        rho_k^i = sqrt((N^i-N_u)^2 + (E^i-E_u)^2 + (U^i-U_u)^2) + c*dt_u
        """
        N_u, E_u, _, _, clock_offset = state
        U_u = 0.0  # Assuming user is at ground level (#TODO)
        
        predicted_ranges = np.zeros(n_stations)
        
        for i in range(n_stations):
            E_i, N_i, U_i = base_stations[i]
            
            # Geometric range
            geometric_range = np.sqrt((N_i - N_u)**2 + (E_i - E_u)**2 + (U_i - U_u)**2)
            
            # Add clock offset
            predicted_ranges[i] = geometric_range + clock_offset
        
        return predicted_ranges

    def compute_jacobian_A(state, base_stations):
        """
        Compute Jacobian matrix A_k (partial derivatives of measurement function)
        A is (n_stations x 5) matrix
        """
        N_u, E_u, _, _, _ = state
        U_u = 0.0  # Assuming user is at ground level
        
        A = np.zeros((n_stations, 5))
        
        for i in range(n_stations):
            E_i, N_i, U_i = base_stations[i]
            
            # Geometric range
            range_i = np.sqrt((N_i - N_u)**2 + (E_i - E_u)**2 + (U_i - U_u)**2)
            
            # Partial derivatives
            if range_i > 1e-10:  # Avoid division by zero
                # ∂ρ/∂N_u = -(N_i - N_u) / range_i
                A[i, 0] = -(N_i - N_u) / range_i  # e_N
                # ∂ρ/∂E_u = -(E_i - E_u) / range_i  
                A[i, 1] = -(E_i - E_u) / range_i  # e_E
                # ∂ρ/∂V_N = 0
                A[i, 2] = 0.0
                # ∂ρ/∂V_E = 0
                A[i, 3] = 0.0
                # ∂ρ/∂(c*dt_u) = 1
                A[i, 4] = 1.0
        
        return A
    
    # Storage for results
    estimated_states = np.zeros((n_measurements, 5))
    covariances = np.zeros((n_measurements, 5, 5))

    # EKF Main Loop
    for k in range(n_measurements):
        # Current measurements
        y_k = P[k, :]
        
        # PREDICTION STEP
        # Predict state: x̂(k|k-1) = T_k * x̂(k-1|k-1)
        if k == 0:
            x_hat_pred = x_hat  # Use initial state for first measurement
        else:
            x_hat_pred = T @ x_hat

        # Predict error covariance: Ĉ(k|k-1) = T_k * Ĉ(k-1|k-1) * T_k^T + C_k^ε
        C_error_pred = T @ C_error @ T.T + C_epsilon

        # MEASUREMENT STEP
        # Compute predicted pseudoranges
        rho_hat = compute_predicted_pseudoranges(x_hat_pred, base_stations)

        # Compute Jacobian matrix
        A = compute_jacobian_A(x_hat_pred, base_stations)

        # Compute Kalman gain: K_k = Ĉ(k|k-1) * A^T * (A * Ĉ(k|k-1) * A^T + C_k^ν)^-1
        S = A @ C_error_pred @ A.T + C_nu
        K = C_error_pred @ A.T @ np.linalg.inv(S)

        # Update state estimate: x̂(k|k) = x̂(k|k-1) + K_k * (y_k - ρ̂_k)
        x_hat = x_hat_pred + K @ (y_k - rho_hat)

        # Update error covariance: Ĉ(k|k) = (I - K_k * A) * Ĉ(k|k-1)
        C_error = (np.eye(5) - K @ A) @ C_error_pred

        # Store results
        estimated_states[k, :] = x_hat
        covariances[k, :, :] = C_error

    return estimated_states, covariances

