import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ex_kalman = pd.read_csv("ex_kalman_LC.csv").to_numpy()
kalman = pd.read_csv("kalman_LC.csv").to_numpy()
UE = pd.read_csv("UE_LC.csv").to_numpy()
base_stations = pd.read_csv("base_stations_LC.csv").to_numpy()

plt.figure(figsize=(12, 6))
plt.scatter(base_stations[:, 0], base_stations[:, 1], c='red', s=100, marker='^', 
           label='Base stations', edgecolors='black')
for i, (E, N, U) in enumerate(base_stations):
    plt.annotate(f'BS{i+1}', (E, N), xytext=(5, 5), textcoords='offset points')
plt.plot(UE[:, 0], UE[:, 1], label='User Equipment', color='orange', marker='o', markersize=5)
plt.plot(kalman[:, 0], kalman[:, 1], label='Kalman Filter', color='red', linewidth=2)
plt.plot(ex_kalman[:, 0], ex_kalman[:, 1], label='Extended Kalman Filter', color='blue', linewidth=2)
plt.scatter(ex_kalman[0, 0], ex_kalman[0, 1], color='green', label='Start Point', s=100)
plt.title('Comparison of Extended Kalman Filter and Kalman Filter')
plt.xlabel('East (m)')
plt.ylabel('North (m)')
plt.legend()
plt.grid()
plt.axis('equal')
plt.tight_layout()
plt.savefig("kalman_comparison.png")
plt.show()
