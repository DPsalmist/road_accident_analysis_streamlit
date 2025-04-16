import streamlit as st
import pandas as pd

# Define the URL for the data file in the GitHub repository
DATA_URL = "datasets/dft-road-casualty-statistics-collision-2023.csv"

@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    return df

df_collision = load_data(DATA_URL)

st.title("High-Risk Road Intersections")
st.subheader("Based on Accident Frequency (2023 Data)")

# Filter for junction accidents
junction_types_to_include = [3, 6, 1, 8, 7, 2, 9]
df_junction_accidents = df_collision[df_collision['junction_detail'].isin(junction_types_to_include)].copy()

# Remove rows with missing latitude or longitude
df_junction_accidents_cleaned = df_junction_accidents.dropna(subset=['latitude', 'longitude']).copy()

# Round latitude and longitude to 4 decimal places
df_junction_accidents_cleaned['rounded_location'] = df_junction_accidents_cleaned['latitude'].round(4).astype(str) + ', ' + df_junction_accidents_cleaned['longitude'].round(4).astype(str)

# Group by the rounded location and count the number of accidents
intersection_accident_counts_v2 = df_junction_accidents_cleaned.groupby('rounded_location').size().sort_values(ascending=False).reset_index(name='accident_frequency')

# Split the rounded_location back into latitude and longitude
intersection_accident_counts_v2[['latitude', 'longitude']] = intersection_accident_counts_v2['rounded_location'].str.split(', ', expand=True).astype(float)

# Scale the accident frequency for marker size
max_freq = intersection_accident_counts_v2['accident_frequency'].max()
intersection_accident_counts_v2['marker_size'] = (intersection_accident_counts_v2['accident_frequency'] / max_freq).tolist() # Convert to list

# Create lists for latitude and longitude
map_lat = intersection_accident_counts_v2['latitude'].head(50).tolist()
map_lon = intersection_accident_counts_v2['longitude'].head(50).tolist()
map_size = intersection_accident_counts_v2['marker_size'][:50]

map_data = pd.DataFrame({'latitude': map_lat, 'longitude': map_lon, 'size': map_size})

st.map(
    map_data[['latitude', 'longitude']],
    #size=map_data['size']
)

# Display top intersections in table
st.write("🚦 Top 10 High-Risk Intersections:")
st.dataframe(map_data[['rounded_location', 'accident_frequency']].head(10))
st.markdown("### Map of Top 50 High-Risk Intersections by Frequency")