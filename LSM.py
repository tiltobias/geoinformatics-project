import numpy as np
import pandas as pd

dt = pd.read_csv("clock_offset.csv").to_numpy()
pr = pd.read_csv("pseudoranges.csv").to_numpy()
BS = pd.read_csv("base_stations.csv").to_numpy()

