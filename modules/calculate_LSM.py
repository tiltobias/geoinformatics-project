import numpy as np

def calculate_LSM(P: np.ndarray, base_stations: np.ndarray, x0=np.zeros(3), max_iter=50, tol=1e-6) -> np.ndarray:
    """
    Least Squares Method for GNSS Positioning

    Args:
        P (np.ndarray): Measured pseudoranges
        base_stations (np.ndarray): Array of base station positions in Local Cartesian coordinates
        x0 (np.ndarray): Initial position estimate
        max_iter (int): Maximum number of iterations
        tol (float): Convergence tolerance

    Returns:
        np.ndarray: Estimated user position. Local Cartesian coordinates (E, N, U, t).
    """
    user_position = np.zeros((P.shape[0], 4))  # Initialize user position array
    user_position[0, :] = np.append(x0, 0)

    def A(x: np.ndarray, base_stations: np.ndarray) -> np.ndarray:
            # Calculate the Jacobian matrix A
            A = np.zeros((base_stations.shape[0], 4))
            for i in range(base_stations.shape[0]):
                x_2 = x[:3]
                x_2[2] = 0
                dist = np.linalg.norm(x_2 - base_stations[i, :])
                A[i, :3] = (x_2 - base_stations[i, :]) / dist
                A[i, 3] = 1
            return A
        
    def dP(x: np.ndarray, base_stations: np.ndarray, epoch: int) -> np.ndarray:
        # Calculate the difference between the measured and calculated pseudoranges
        dP = np.zeros(base_stations.shape[0])
        x_2 = x[:3]
        x_2[2] = 0 
        for i in range(base_stations.shape[0]):
            dist = np.linalg.norm(x_2[:3] - base_stations[i, :])
            dP[i] = P[epoch, i] - dist # P[0, i] is the measured pseudorange for epoch 0
        return dP

    for epoch in range(P.shape[0]):
        
        x = user_position[epoch, :].copy()

        
        for i in range(max_iter):
            X_old = user_position[epoch, :]
            A_ = A(X_old, base_stations)
            dPrs_ = dP(X_old, base_stations, epoch)
            X_hat, *_ = np.linalg.lstsq(A_, dPrs_, rcond=None) 

            dx, *_ = np.linalg.lstsq(A(x, base_stations), dP(x, base_stations, epoch), rcond=None)
            c = 299792458
            x = np.append(x[0:3] + dx[0:3], (np.array([x[3] + dx[3]]))/c)

            if np.linalg.norm(dx) < tol:
                print('Break at cycle: ', i+1, ' epoch: ', epoch+1, ' with position: ', x)
                break
            
        user_position[epoch, :] = x
        if epoch + 1 < P.shape[0]:
            user_position[epoch + 1] = x

    return user_position