import streamlit as st
import pandas as pd
import pydeck as pdk

# Define the path to your data file
DATA_URL = "datasets/dft-road-casualty-statistics-collision-2023.csv"

# Cache the data load for performance
@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    return df

# Load the collision data
df_collision = load_data(DATA_URL)

# App title and subtitle
st.title("High-Risk Road Intersections")
st.subheader("Based on Road Accident Frequency (2023 UK Data)")

# Filter for relevant junction types (based on dataset coding)
junction_types_to_include = [1, 2, 3, 6, 7, 8, 9]  # Examples: T-junction, crossroads, etc.
df_junction_accidents = df_collision[df_collision['junction_detail'].isin(junction_types_to_include)].copy()

# Drop rows without latitude or longitude
df_junction_accidents_cleaned = df_junction_accidents.dropna(subset=['latitude', 'longitude']).copy()

# Create a rounded location string to group nearby accidents
df_junction_accidents_cleaned['rounded_location'] = (
    df_junction_accidents_cleaned['latitude'].round(4).astype(str) + ', ' +
    df_junction_accidents_cleaned['longitude'].round(4).astype(str)
)

# Group by rounded location and count accident frequency
intersection_accident_counts = (
    df_junction_accidents_cleaned
    .groupby('rounded_location')
    .size()
    .sort_values(ascending=False)
    .reset_index(name='accident_frequency')
)

# Extract lat/lon from the rounded string
intersection_accident_counts[['latitude', 'longitude']] = intersection_accident_counts['rounded_location'].str.split(', ', expand=True).astype(float)

# Normalize for marker size
max_freq = intersection_accident_counts['accident_frequency'].max()
intersection_accident_counts['marker_size'] = (intersection_accident_counts['accident_frequency'] / max_freq) * 100

# Limit to top 50 intersections
map_data = intersection_accident_counts.head(50)

# Define pydeck scatterplot layer
layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_data,
    get_position='[longitude, latitude]',
    get_radius='marker_size * 3000',
    get_fill_color='[255, 100, 100, 160]',
    pickable=True,
    auto_highlight=True,
)

# Define initial map view state
view_state = pdk.ViewState(
    latitude=map_data['latitude'].mean(),
    longitude=map_data['longitude'].mean(),
    zoom=10,
    pitch=30,
)

# Render the pydeck map
st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "Lat: {latitude}\nLon: {longitude}\nAccidents: {accident_frequency}"},
))

# Display top intersections in table
st.write("🚦 Top 10 High-Risk Intersections:")
st.dataframe(map_data[['rounded_location', 'accident_frequency']].head(10))
st.markdown("### Map of Top 50 High-Risk Intersections by Frequency")