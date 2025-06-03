import numpy as np
import pandas as pd
from transformations import X_GC_GG, R_GC_LC

P = pd.read_csv("pseudoranges.csv").to_numpy()
base_stations = pd.read_csv("base_stations_LC.csv").to_numpy()

def LSM(P: np.ndarray, x0: np.ndarray, base_stations: np.ndarray, max_iter=50, tol=1e-6) -> np.ndarray:
    
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

        # b = P - dPrs_
        # Prs = A_ @ x + b
        # v = P - Prs
        # sigma0_squared = v.T @ v / (8 - 4) # m - n, where m is the number of pseudoranges and n is the number of unknowns (4: E, N, U, t)
        # C_xx = sigma0_squared * np.linalg.inv(A_.T @ A_)
        # # Calculation of PDOP
        # Q_geom = np.linalg.inv(A_.T @ A_)[0:3, 0:3]
        # R_0 = R_GC_LC(X_GC_GG(x)) # flatten the last element of X_history for R calculation
        # Q_LC = R_0 @ Q_geom @ R_0.T
        # PDOP = np.sqrt(np.trace(Q_LC)) # trace = sum of diagonal elements
        # print(np.diag(C_xx))


    return user_position

    # print(user_position)

    # # Initial guess of user position
    # x_tilde = x0.copy()

    
x0 = base_stations.mean(axis=0)
print("Initial guess of user position: ", x0)
x0_2 = np.append(x0[0:2], 0)
print("Initial guess of user position (2D): ", x0_2)

test = LSM(P, x0, base_stations)
    
print(test)

# Write the user position to a CSV file
user_position_df = pd.DataFrame(test, columns=["E", "N", "U", "t"])
user_position_df.to_csv('./user_position_LSM.csv', index=False)

# plot clock offset
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(test[:, 3], label='Clock Offset (m)')
plt.title('Clock Offset Over Time')
plt.xlabel('Epoch')
plt.ylabel('Clock Offset (m)')
plt.grid()
plt.legend()
plt.show()
