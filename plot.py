import pandas as pd
import folium

df = pd.read_csv('./KinematicData/T1.csv')

# Create a map centered around the average latitude and longitude
map = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=19, max_zoom=30)

# Add a circle marker for each row in the DataFrame
for idx, row in df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']], radius=5,
        color='red', fill=False, fill_opacity=0.6,
        popup=folium.Popup(f"Time: {row['Local_time']}", parse_html=True)
    ).add_to(map)


# Make a line between the points
folium.PolyLine(
    locations=df[['Latitude', 'Longitude']].values,
    color='blue', weight=2.5, opacity=1
).add_to(map)

map.save('map.html')