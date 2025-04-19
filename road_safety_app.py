import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------
# 🚚 1. Load the Data
# -------------------------
DATA_URL = "datasets/dft-road-casualty-statistics-collision-2023.csv"

@st.cache_data
def load_data(url):
    df = pd.read_csv(url, low_memory=False)
    return df

df_collision = load_data(DATA_URL)

# -------------------------
# 🧭 App Title
# -------------------------
st.title("High-Risk Road Intersections in the UK")
st.subheader("Based on 2023 Road Collision Data")

st.markdown("""
**🔍 Project Overview:**  
This interactive dashboard helps identify the **top high-risk road intersections** in the UK, based on reported accidents from the 2023 road casualty data.  
You can filter accidents by **region, junction type, severity**, and **time**.  
It’s designed to support non-technical users like city planners, safety teams, or policymakers in visualizing and prioritizing intervention zones.
""")

st.markdown("Use the filters on the left sidebar to refine results and explore data more interactively.")

# -------------------------
# 🎛️ 2. Sidebar Filters
# -------------------------
st.sidebar.header("🔎 Filters")

# Junction Type Filter
st.sidebar.markdown("### Junction Type")
junction_type_labels = {
    0: "Not at junction",
    1: "Roundabout",
    2: "Mini-roundabout",
    3: "T or staggered junction",
    5: "Slip road",
    6: "Crossroads",
    7: "More than 4 arms (not roundabout)",
    8: "Private drive or entrance",
    9: "Other junction"
}
available_junctions = sorted(df_collision['junction_detail'].dropna().unique())
selected_junctions = st.sidebar.multiselect(
    "Select Junction Types",
    options=available_junctions,
    default=[1, 2, 3, 6, 7, 8, 9],
    format_func=lambda x: junction_type_labels.get(x, str(x))
)

# Region Filter
st.sidebar.markdown("### Region")
available_regions = sorted(df_collision['local_authority_ons_district'].dropna().unique())
selected_regions = st.sidebar.multiselect(
    "Select Regions (ONS Districts)",
    options=available_regions,
    default=available_regions[:5]
)

# Accident Severity Filter
st.sidebar.markdown("### Severity")
severity_map = {1: "Fatal", 2: "Serious", 3: "Slight"}
available_severities = df_collision['accident_severity'].dropna().unique()
selected_severities = st.sidebar.multiselect(
    "Select Severity Level",
    options=available_severities,
    default=available_severities,
    format_func=lambda x: severity_map.get(x, str(x))
)

# Time Filter (Month)
st.sidebar.markdown("### Month")
df_collision['month'] = pd.to_datetime(df_collision['date'], errors='coerce').dt.month
months_map = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}
available_months = sorted(df_collision['month'].dropna().unique())
selected_months = st.sidebar.multiselect(
    "Select Month(s)",
    options=available_months,
    default=available_months,
    format_func=lambda x: months_map.get(x, f"Month {x}")
)

# -------------------------
# 🔍 3. Apply Filters
# -------------------------
df_filtered = df_collision[
    df_collision['junction_detail'].isin(selected_junctions) &
    df_collision['local_authority_ons_district'].isin(selected_regions) &
    df_collision['accident_severity'].isin(selected_severities) &
    df_collision['month'].isin(selected_months)
].copy()

df_filtered = df_filtered.dropna(subset=['latitude', 'longitude'])

df_filtered['rounded_location'] = (
    df_filtered['latitude'].round(4).astype(str) + ', ' +
    df_filtered['longitude'].round(4).astype(str)
)

# -------------------------
# 📊 4. Aggregate Accident Data
# -------------------------
intersection_accident_counts = (
    df_filtered.groupby('rounded_location')
    .size()
    .sort_values(ascending=False)
    .reset_index(name='accident_frequency')
)

intersection_accident_counts[['latitude', 'longitude']] = (
    intersection_accident_counts['rounded_location']
    .str.split(', ', expand=True).astype(float)
)

max_freq = intersection_accident_counts['accident_frequency'].max()
intersection_accident_counts['marker_size'] = intersection_accident_counts['accident_frequency'] / max_freq

# -------------------------
# 🗺️ 5. Map Display (Top 50)
# -------------------------
st.markdown("### 🗺️ Map of Top 50 High-Risk Intersections")
map_data = intersection_accident_counts[['latitude', 'longitude']].head(50)
st.map(map_data)

# -------------------------
# 📋 6. Data Table Display
# -------------------------
st.markdown("### 🚦 Top 10 Intersections with Highest Accident Frequency")

display_df = intersection_accident_counts[['rounded_location', 'latitude', 'longitude', 'accident_frequency']].head(10)
display_df.columns = ['Location (Lat, Lon)', 'Latitude', 'Longitude', 'Accident Count']
st.dataframe(display_df)

# -------------------------
# 📊 7. Bar Chart
# -------------------------
st.markdown("### 📊 Accident Frequency by Intersection (Bar Chart)")

fig_bar = px.bar(
    display_df,
    x='Location (Lat, Lon)',
    y='Accident Count',
    color='Accident Count',
    color_continuous_scale='Reds',
    title='Top 10 Intersections with Highest Accident Frequency',
    labels={'Accident Count': 'Accidents'},
    height=400
)
fig_bar.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig_bar, use_container_width=True)

# -------------------------
# 🥧 8. Pie Chart: Severity Breakdown
# -------------------------
st.markdown("### 🥧 Accident Severity Breakdown")

severity_distribution = df_filtered['accident_severity'].map(severity_map).value_counts().reset_index()
severity_distribution.columns = ['Severity', 'Count']

fig_pie = px.pie(
    severity_distribution,
    names='Severity',
    values='Count',
    title='Accident Severity Distribution in Filtered Data',
    color_discrete_sequence=px.colors.sequential.RdBu
)
st.plotly_chart(fig_pie, use_container_width=True)

# -------------------------
# 🧠 9. Key Observations
# -------------------------
st.markdown("### 🧠 Insights & Observations")

most_accident_prone = display_df.iloc[0]
st.markdown(f"""
- 🚨 The intersection at **{most_accident_prone['Location (Lat, Lon)']}** recorded the **highest number of accidents**: **{most_accident_prone['Accident Count']}**.
- 📌 The top 10 intersections collectively account for **{display_df['Accident Count'].sum()}** reported accidents in the selected filters.
- 🧾 Filter applied: **{len(df_filtered)}** accidents matched your criteria.
""")

# -------------------------
# 💾 10. Export Options
# -------------------------
st.markdown("### 📤 Download Table")

csv_download = display_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download as CSV",
    data=csv_download,
    file_name="top_10_intersections.csv",
    mime="text/csv"
)

# Optional: Download as image if dataframe-image is available
try:
    import dataframe_image as dfi
    import tempfile
    temp_img_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    dfi.export(display_df, temp_img_path)
    with open(temp_img_path, "rb") as img_file:
        st.download_button(
            label="Download as Image (PDF Alternative)",
            data=img_file,
            file_name="top_10_intersections.png",
            mime="image/png"
        )
except ImportError:
    st.info("To enable image export, install `dataframe-image`: `pip install dataframe-image`")

# -------------------------
# ✅ End of App
# -------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and UK Road Safety Data (DfT, 2023)")
