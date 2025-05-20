import numpy as np
import pandas as pd

P = pd.read_csv("pseudoranges.csv").to_numpy()
base_stations = pd.read_csv("base_stations_LC.csv").to_numpy()

# Initialization
x0 = base_stations.mean(axis=0)
print("Initial guess of user position: ", x0)
x0 = np.append(x0[0:2], 0)
print("Initial guess of user position (2D): ", x0)
c = 299792458  # Speed of light in m/s
max_iter = 100

def rho_tilde(x, base_stations):
    return np.array([np.linalg.norm(x[:3] - bs) for bs in base_stations])

def A(x, base_stations):
    """
    Calculate the Jacobian matrix A for the given user position x and base stations.
    """
    A_ = np.zeros((base_stations.shape[0], 4))
    rho_tilde_ = rho_tilde(x, base_stations)
    for i in range(base_stations.shape[0]):
        A_[i, 0] = (x[0] - base_stations[i, 0]) / rho_tilde_[i]
        A_[i, 1] = (x[1] - base_stations[i, 1]) / rho_tilde_[i]
        A_[i, 2] = (x[2] - base_stations[i, 2]) / rho_tilde_[i]
        A_[i, 3] = 1
    return A_

def b(x, base_stations): 
    return rho_tilde(x, base_stations)

for k in range(max_iter): # max_iter

    A_ = A(x0, base_stations)
    b_ = b(x0, base_stations)
    print(np.linalg.matrix_rank(A_))
    N = A_.T @ A_
    N_inv = np.linalg.inv(N)
    dP_r_s = P[0, :] - b(x0, base_stations)

    est_corr = N_inv @ A_.T @ dP_r_s

    x0 += np.append(est_corr[0:3], est_corr[3]/c) 

    thresh = 1e-3

    if est_corr[0:3].max() < thresh:
        print("Convergence reached, broke at iteration: ", k)
        break
    elif k == max_iter - 1:
        print("Max iterations reached, broke at iteration: ", k)




    
user_position = np.zeros((P.shape[0], 4))  # Initialize user position array
user_position[0, :] = np.append(x0, 0)