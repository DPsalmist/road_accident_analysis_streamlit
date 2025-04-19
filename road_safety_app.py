import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------
# 🚚 1. Load the Data
# -------------------------
DATA_URL_COLLISIONS = "datasets/dft-road-casualty-statistics-collision-2023.csv"
DATA_URL_CASUALTIES = "datasets/dft-road-casualty-statistics-casualty-2023.csv"
DATA_URL_VEHICLES = "datasets/dft-road-casualty-statistics-vehicle-2023.csv"

@st.cache_data
def load_data(url):
    df = pd.read_csv(url, low_memory=False)
    return df

df_collision = load_data(DATA_URL_COLLISIONS)
df_casualties = load_data(DATA_URL_CASUALTIES)
df_vehicles = load_data(DATA_URL_VEHICLES)

# -------------------------
# ⚙️ 2. Merge DataFrames
# -------------------------
df_merged = pd.merge(df_collision, df_casualties, on='accident_index', how='inner')
df_merged = pd.merge(df_merged, df_vehicles, on='accident_index', how='inner')

# -------------------------
# 🧭 App Title (Initial Part - Rest of UI will be in tabs)
# -------------------------
st.title("UK Road Accident Analysis")
st.subheader("Based on 2023 Data")

st.markdown("Use the filters on the left sidebar to explore the data.")

# -------------------------
# 🎛️ 3. Sidebar Filters (Keep existing filters)
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
st.sidebar.markdown("### Severity (for initial exploration)")
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
# 🔍 4. Apply Initial Filters (for the existing tabs)
# -------------------------
df_filtered = df_merged[
    df_merged['junction_detail'].isin(selected_junctions) &
    df_merged['local_authority_ons_district'].isin(selected_regions) &
    df_merged['accident_severity_x'].isin(selected_severities) & # Use '_x' for collisions severity
    df_merged['month'].isin(selected_months)
].copy()

df_filtered = df_filtered.dropna(subset=['latitude', 'longitude'])

df_filtered['rounded_location'] = (
    df_filtered['latitude'].round(4).astype(str) + ', ' +
    df_filtered['longitude'].round(4).astype(str)
)

# -------------------------
# 📊 5. Aggregate Accident Data (for existing tabs)
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

# --- Add Top N Slider ---
top_n = st.sidebar.slider("Number of Top Intersections to Display", 1, 100, 10)

# -------------------------
#  Tabbed Interface (Add a new tab for ML)
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📍 Map",
    "📊 Data & Stats",
    "Insights",
    "Download",
    "About",
    "🧠 Severity Prediction"
])

with tab1:
    # 🗺️ Map Display (Dynamic Top N)
    st.markdown("### 🗺️ Map of Top High-Risk Intersections")
    map_data = intersection_accident_counts[['latitude', 'longitude']].head(top_n)
    st.map(map_data)

with tab2:
    # 📋 Data Table Display
    st.markdown("### 🚦 Top Intersections with Highest Accident Frequency")
    display_df = intersection_accident_counts[['rounded_location', 'latitude', 'longitude', 'accident_frequency']].head(top_n)
    display_df.columns = ['Location (Lat, Lon)', 'Latitude', 'Longitude', 'Accident Count']
    st.dataframe(display_df)

    # 📊 Bar Chart
    st.markdown("### 📊 Accident Frequency by Intersection (Bar Chart)")
    fig_bar = px.bar(
        display_df,
        x='Location (Lat, Lon)',
        y='Accident Count',
        color='Accident Count',
        color_continuous_scale='Reds',
        title=f'Top {top_n} Intersections with Highest Accident Frequency',
        labels={'Accident Count': 'Accidents'},
        height=400
    )
    fig_bar.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True)

    # 🥧 Pie Chart: Severity Breakdown
    st.markdown("### 🥧 Accident Severity Breakdown")
    severity_distribution = df_filtered['accident_severity_x'].map(severity_map).value_counts().reset_index() # Use '_x' for collisions severity
    severity_distribution.columns = ['Severity', 'Count']
    fig_pie = px.pie(
        severity_distribution,
        names='Severity',
        values='Count',
        title='Accident Severity Distribution in Filtered Data',
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with tab3:
    # 🧠 Key Observations
    st.markdown("### 🧠 Insights & Observations")
    if not display_df.empty:
        most_accident_prone = display_df.iloc[0]
        st.markdown(f"""
    - 🚨 The intersection at **{most_accident_prone['Location (Lat, Lon)']}** recorded the **highest number of accidents**: **{most_accident_prone['Accident Count']}**.
    - 📌 The top {top_n} intersections collectively account for **{display_df['Accident Count'].sum()}** reported accidents in the selected filters.
    """)
    st.markdown(f"- 🧾 Filter applied: **{len(df_filtered)}** accidents matched your criteria.")

with tab4:
    # 💾 Export Options
    st.markdown("### 📤 Download Top Intersections Data")
    if not display_df.empty:
        csv_download = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download as CSV",
            data=csv_download,
            file_name=f"top_{top_n}_intersections.csv",
            mime="text/csv"
        )
    else:
        st.warning("No top intersections to download based on the current filters.")

with tab5:
    # About Section
    st.markdown("### ℹ️ About This Dashboard")
    st.markdown("""
    This dashboard visualizes road accident data from the UK for the year 2023, focusing on identifying high-risk intersections.
    It allows users to filter the data by various criteria such as junction type, region, accident severity, and month.
    The goal is to provide an accessible tool for understanding accident hotspots and supporting road safety initiatives.
    """)
    st.caption("Data Source: UK Road Safety Data (Department for Transport, 2023)")
    st.caption("Built with ❤️ using Streamlit, Pandas, and Plotly.")

with tab6:
    st.markdown("### 🧠 Accident Severity Prediction")
    st.markdown("This section will allow you to input accident characteristics to predict the severity.")
    st.markdown("We will build the prediction model and then add input fields here.")

# -------------------------
# ✅ End of App
# -------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and UK Road Safety Data (DfT, 2023)")