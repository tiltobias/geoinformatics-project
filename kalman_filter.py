import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('./user_position_LSM.csv')[['E', 'N']].to_numpy()

# data = np.genfromtxt('./user_position_LSM.csv', delimiter=',')
# print("Loaded data: \n", data)

def y(t: int) -> np.ndarray:
    # Returns observation at epoch t
    return np.array([[data[t, 0]], [data[t, 1]]])

X, Y = data[0, 0], data[0, 1]
X_dot, Y_dot = 0, 0
x = np.array([[[X], [Y], [X_dot], [Y_dot]]])

C_error = np.diag([10**2, 10**2, 1, 1])
C_model = np.diag([1, 1, 1, 1])
C_obs = np.diag([5**2, 5**2])  # Adjust smoothness of the path ======================================
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

plt.plot(data[:, 0], data[:, 1], "b.", markersize=3) # all data
plt.plot(x[:, 0, 0], x[:, 1, 0], "r-", linewidth=3) # estimated path
plt.plot(data[0, 0], data[0, 1], "go") # start point
plt.legend(["Data", "Estimated path", "Start point"])
plt.show()


# Save estimated positions to CSV
results_df = pd.DataFrame(x[:, :, 0], columns=['E', 'N', 'VN', 'VE'])
results_df.to_csv("kalman_LC.csv", index=False)