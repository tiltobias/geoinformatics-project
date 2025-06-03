import pandas as pd
import folium

df = pd.read_csv('./KinematicData/T1.csv')
df_lsm = pd.read_csv('./user_position_GG.csv')
df_kalman = pd.read_csv('./kalman_GG.csv')
df_ex_kalman = pd.read_csv('./ex_kalman_GG.csv')


# Create a map centered around the average latitude and longitude
map = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=19, max_zoom=30)

# Add a circle marker for each row in the DataFrame


for idx, row in df_lsm.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=folium.Popup("LSM", parse_html=True),
        icon=folium.Icon(color='red')
    ).add_to(map)

for idx, row in df_kalman.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=folium.Popup("Kalman", parse_html=True),
        icon=folium.Icon(color='blue')
    ).add_to(map)

for idx, row in df_ex_kalman.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=folium.Popup("Ex Kalman", parse_html=True),
        icon=folium.Icon(color='orange')
    ).add_to(map)

for idx, row in df.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=folium.Popup("Measured", parse_html=True),
        icon=folium.Icon(color='green')
    ).add_to(map)


# Make a line between the points
# folium.PolyLine(
#     locations=df[['Latitude', 'Longitude']].values,
#     color='blue', weight=2.5, opacity=1
# ).add_to(map)

map.save('map.html')