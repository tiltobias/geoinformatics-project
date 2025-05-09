import numpy as np
import pandas as pd

df = pd.read_csv('./KinematicData/T1.csv')
mean_ue_height = np.mean(df['Height'])
height_base_stations = mean_ue_height

base_stations = [
    np.array(["BS1",45.3435, 9.0102, height_base_stations]), # bottom left corner BS1
    np.array(["BS2",45.3435, 9.0149, height_base_stations]), # bottom right corner BS2
    np.array(["BS3",45.3451, 9.0149, height_base_stations]), # top right corner BS3
    np.array(["BS4",45.3451, 9.0102, height_base_stations]), # top left corner BS4
    np.array(["BS5",45.3438, 9.0109, height_base_stations]), # left roundabout BS5
    np.array(["BS6",45.3446, 9.0137, height_base_stations]), # right roundabout BS6
    np.array(["BS7",45.3446, 9.0126, height_base_stations]), # middle top BS7
    np.array(["BS8",45.3436, 9.0125, height_base_stations]), # middle bottom BS8
]

# Save the base stations to a CSV file
base_stations_df = pd.DataFrame(base_stations, columns=["Title","Latitude", "Longitude", "Height"])
base_stations_df.to_csv('./base_stations.csv', index=False)

