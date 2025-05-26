import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)  # Seed to reproduce results
n = 1000
clock_offset_base = 1000 # 1 microsecond
clock_offset_change = .5 # 0.5 nanoseconds

# Randomly assign one or zero to n values -> go up or down
up_or_down = np.random.randint(0, 2, size=n)

# Array which holds clock offsets
clock_offset = np.zeros(n) + clock_offset_base

# Loop which determines random walk path
for i in range(1, n): # start at 1 to avoid index error
    # Walk up if 1 - increment by clock offset change
    if up_or_down[i] == 1:
        clock_offset[i] = clock_offset[i - 1] + clock_offset_change
    # Walk down if 0
    else:
        clock_offset[i] = clock_offset[i - 1] - clock_offset_change


# Write random walk clock offsets to csv 
df = pd.DataFrame(clock_offset, columns=["Clock Offset"])
df.to_csv("clock_offset.csv", index=False)


# Plot walk
plt.plot(clock_offset)
plt.title("Clock Offset")
plt.xlabel("Time")
plt.ylabel("Offset")
plt.grid()
plt.show()