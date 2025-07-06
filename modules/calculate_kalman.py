import numpy as np

def calculate_kalman(LSM_coords: np.ndarray, sigma_obs=5., sigma_error=10.) -> np.ndarray:
    """
    Calculate the Kalman filter for the given data.
    
    Args:
        LSM_coords (np.ndarray): Input data containing position information. Local Cartesian coordinates (E, N).
        sigma_obs (float): Standard deviation of the observation noise.
        sigma_error (float): Standard deviation of the model error.
        
    Returns:
        np.ndarray: Estimated positions after applying the Kalman filter. Local Cartesian coordinates (E, N).
    """
    data = LSM_coords

    def y(t: int) -> np.ndarray:
        # Returns observation at epoch t
        return np.array([[data[t, 0]], [data[t, 1]]])

    X, Y = data[0, 0], data[0, 1]
    X_dot, Y_dot = 0, 0
    x = np.array([[[X], [Y], [X_dot], [Y_dot]]])

    C_error = np.diag([sigma_error**2, sigma_error**2, 1, 1])
    C_model = np.diag([1, 1, 1, 1])
    C_obs = np.diag([sigma_obs**2, sigma_obs**2])  # Adjust smoothness of the path ======================================
    I = np.identity(4)

    T = np.array([[1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

    A = np.array([[1, 0, 1, 0], 
                [0, 1, 0, 1]])

    x_hat = x[0] 

    for t in range(len(data)):
        x_tilde = T @ x_hat
        
        K = C_model + T @ C_error @ T.T 
        G = K @ A.T @ np.linalg.inv(A @ K @ A.T + C_obs) 
        x_hat = (I - G @ A) @ x_tilde + G @ y(t)
        C_error = (I - G @ A) @ K

        x = np.vstack((x, [x_hat]))
        
    return x
