import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000
clock_offset_base = 1000
clock_offset_change = 0.5

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



# Plot walk
import matplotlib.pyplot as plt
plt.plot(clock_offset)
plt.title("Clock Offset")
plt.xlabel("Time")
plt.ylabel("Offset")
plt.grid()
plt.show()

df = pd.DataFrame(clock_offset, columns=["Clock Offset"])
df.to_csv("clock_offset.csv", index=False)