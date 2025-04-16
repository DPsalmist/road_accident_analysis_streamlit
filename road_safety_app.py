import streamlit as st
import pandas as pd

# Load the Collisions data (replace with the correct path if needed in Streamlit Cloud)
# Assuming the CSV file is in the same repository
DATA_URL = "datasets/dft-road-casualty-statistics-collision-2023.csv"
df_collision = pd.read_csv(DATA_URL)

print(df_collision.head())

# # Filter for junction accidents
# junction_types_to_include = [3, 6, 1, 8, 7, 2, 9]
# df_junction_accidents = df_collision[df_collision['junction_detail'].isin(junction_types_to_include)].copy()

# # Remove rows with missing latitude or longitude
# df_junction_accidents_cleaned = df_junction_accidents.dropna(subset=['latitude', 'longitude']).copy()

# # Round latitude and longitude to 4 decimal places
# df_junction_accidents_cleaned['rounded_location'] = df_junction_accidents_cleaned['latitude'].round(4).astype(str) + ', ' + df_junction_accidents_cleaned['longitude'].round(4).astype(str)

# # Group by the rounded location and count the number of accidents
# intersection_accident_counts_v2 = df_junction_accidents_cleaned.groupby('rounded_location').size().sort_values(ascending=False).reset_index(name='accident_frequency')

# # Split the rounded_location back into latitude and longitude
# intersection_accident_counts_v2[['latitude', 'longitude']] = intersection_accident_counts_v2['rounded_location'].str.split(', ', expand=True).astype(float)

# # Create a DataFrame for the map (using 'latitude' and 'longitude' column names)
# map_data = intersection_accident_counts_v2[['latitude', 'longitude', 'accident_frequency']].head(50)

# st.title("High-Risk Road Intersections")
# st.subheader("Based on Accident Frequency (2023 Data)")

# st.map(map_data[['latitude', 'longitude']])

# st.write("Top Potential High-Risk Intersections:")
# st.dataframe(map_data.head())

#######

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
intersection_accident_counts_v2[['latitude_rounded', 'longitude_rounded']] = intersection_accident_counts_v2['rounded_location'].str.split(', ', expand=True).astype(float)

# Create a DataFrame for the map
map_data = intersection_accident_counts_v2[['latitude_rounded', 'longitude_rounded', 'accident_frequency']].head(50) # Limiting to top 50 for initial view

st.title("High-Risk Road Intersections")
st.subheader("Based on Accident Frequency (2023 Data)")

st.map(map_data[['latitude_rounded', 'longitude_rounded']])

st.write("Top Potential High-Risk Intersections:")
st.dataframe(map_data.head())