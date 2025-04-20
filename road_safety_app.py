import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff  # For confusion matrix

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


@st.cache_data
def load_data(url):
    try:
        df = pd.read_csv(url, low_memory=False)
        return df
    except Exception as e:
        st.error(f"Errors loading data from {url}: {e}")
        return None

def accident_severity_prediction_tab(df_merged):
    st.markdown("### 🧠 Accident Severity Prediction")
    st.markdown("This section displays the model evaluation for accident severity prediction.")

    try:
        if df_merged is not None:
            ml_df = df_merged[[
                'vehicle_type',
                'age_of_driver',
                'road_surface_conditions',
                'junction_detail',
                'light_conditions',
                'weather_conditions',
                'speed_limit',
                'accident_severity'
            ]].copy()
            ml_df['accident_severity'] = pd.to_numeric(ml_df['accident_severity'], errors='coerce').astype('Int64')
            ml_df_cleaned = ml_df.dropna().copy() # Use .copy() to avoid SettingWithCopyWarning

            # Merge 99 and -1 in 'junction_detail' to 'Unknown/Missing'
            ml_df_cleaned['junction_detail'] = ml_df_cleaned['junction_detail'].replace([99, -1], 'Unknown/Missing')

            ml_df_encoded = pd.get_dummies(ml_df_cleaned, columns=[
                'vehicle_type',
                'road_surface_conditions',
                'junction_detail',
                'light_conditions',
                'weather_conditions'
            ])
            X = ml_df_encoded.drop('accident_severity', axis=1)
            y = ml_df_encoded['accident_severity'].astype(int)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            model = RandomForestClassifier(n_estimators=20, random_state=42, class_weight='balanced')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.subheader("Model Evaluation - Random Forest Model")
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {accuracy:.2f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            cm_labels = ['Fatal', 'Serious', 'Slight']
            fig_cm = ff.create_annotated_heatmap(cm, x=cm_labels, y=cm_labels, colorscale='Blues')
            fig_cm.update_layout(xaxis_title='Predicted Severity', yaxis_title='Actual Severity', xaxis=dict(side='bottom'), yaxis=dict(autorange='reversed'))
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.warning("Merged data is not available, cannot perform accident severity prediction.")
        st.markdown("---")
    except Exception as e:
        st.error(f"Error occurred during model evaluation: {e}")
        st.info("Please check the logs for more details.")

# Load Data with error handling
DATA_URL_COLLISIONS = "datasets/dft-road-casualty-statistics-collision-2023.csv"
DATA_URL_CASUALTIES = "datasets/dft-road-casualty-statistics-casualty-2023.csv"
DATA_URL_VEHICLES = "datasets/dft-road-casualty-statistics-vehicle-2023.csv"
df_collision = load_data(DATA_URL_COLLISIONS)
df_casualties = load_data(DATA_URL_CASUALTIES)
df_vehicles = load_data(DATA_URL_VEHICLES)

# Merge DataFrames with error handling
df_merged = None
if df_collision is not None and df_casualties is not None and df_vehicles is not None:
    try:
        df_merged = pd.merge(df_collision, df_casualties, on='accident_index', how='inner')
        df_merged = pd.merge(df_merged, df_vehicles, on='accident_index', how='inner')
    except Exception as e:
        st.error(f"Error merging DataFrames: {e}")

# -------------------------
# 🧭 App Title
# -------------------------
st.title("UK Road Accident Analysis")
st.subheader("Based on 2023 Data")
st.markdown("Use the filters on the left sidebar to explore the data.")

# -------------------------
# 🎛️ Sidebar Filters with error handling for invalid junction selections
# -------------------------
st.sidebar.header("🔎 Filters")
selected_junctions = []
selected_regions = []
selected_severities = []
selected_months = []

if df_collision is not None and df_merged is not None:
    try:
        junction_type_labels = {0: "Not at junction", 1: "Roundabout", 2: "Mini-roundabout", 3: "T or staggered junction", 5: "Slip road", 6: "Crossroads", 7: "More than 4 arms (not roundabout)", 8: "Private drive or entrance", 9: "Other junction"}
        available_junctions_raw = sorted(df_collision['junction_detail'].dropna().unique())
        available_junctions = [j for j in available_junctions_raw if j not in [99, -1]]
        selected_junctions = st.sidebar.multiselect(
            "Select Junction Types",
            options=available_junctions,
            default=[1, 2, 3, 6, 7, 8, 9],
            format_func=lambda x: junction_type_labels.get(x, str(x))
        )

        available_regions = sorted(df_collision['local_authority_ons_district'].dropna().unique())
        selected_regions = st.sidebar.multiselect("Select Regions (ONS Districts)", options=available_regions, default=available_regions[:5])

        severity_map = {1: "Fatal", 2: "Serious", 3: "Slight"}
        available_severities = df_collision['accident_severity'].dropna().unique()
        selected_severities = st.sidebar.multiselect("Select Severity Level", options=available_severities, default=available_severities, format_func=lambda x: severity_map.get(x, str(x)))

        df_merged['date'] = pd.to_datetime(df_merged['date'], errors='coerce')
        df_merged['month'] = df_merged['date'].dt.month
        months_map = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
        available_months = sorted(df_merged['month'].dropna().unique())
        selected_months = st.sidebar.multiselect("Select Month(s)", options=available_months, default=available_months, format_func=lambda x: months_map.get(x, f"Month {x}"))
    except Exception as e:
        st.sidebar.error(f"Error creating sidebar filters: {e}")

# -------------------------
# 🔍 Apply Initial Filters
# -------------------------
df_filtered = None
if df_merged is not None:
    try:
        valid_selected_junctions = [j for j in selected_junctions if j in available_junctions]
        df_filtered = df_merged[
            df_merged['junction_detail'].isin(valid_selected_junctions) &
            df_merged['local_authority_ons_district'].isin(selected_regions) &
            df_merged['accident_severity'].isin(selected_severities) &
            df_merged['month'].isin(selected_months)
        ].copy()
        df_filtered = df_filtered.dropna(subset=['latitude', 'longitude'])
        if not df_filtered.empty:
            df_filtered['rounded_location'] = (df_filtered['latitude'].round(4).astype(str) + ', ' + df_filtered['longitude'].round(4).astype(str))
    except Exception as e:
        st.error(f"Error applying filters: {e}")

# -------------------------
# 📊 Aggregate Accident Data
# -------------------------
intersection_accident_counts = pd.DataFrame()
if df_filtered is not None and not df_filtered.empty:
    try:
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
        max_freq = intersection_accident_counts['accident_frequency'].max() if not intersection_accident_counts.empty else 1
        intersection_accident_counts['marker_size'] = intersection_accident_counts['accident_frequency'] / max_freq
    except Exception as e:
        st.error(f"Error aggregating accident data: {e}")

top_n = st.sidebar.slider("Number of Top Intersections to Display", 1, 100, 10)

# -------------------------
#  Tabbed Interface
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "About",
    "📍 Map",
    "📊 Data & Stats",
    "Insights",
    "Download",
    "🧠 Accident Severity Prediction"
])

with tab1:
    try:
        st.markdown("### ℹ️ About This Dashboard")
        st.markdown("""This dashboard visualizes road accident data from the UK for the year 2023, focusing on identifying high-risk intersections. It allows users to filter the data by various criteria such as junction type, region, accident severity, and month. The main objective is to provide an accessible tool for understanding accident hotspots and supporting road safety initiatives.""")
        st.caption("Data Source: UK Road Safety Data (Department for Transport, 2023)")
        st.caption("Built with ❤️ using Streamlit, Pandas, and Plotly.")
    except Exception as e:
        st.error(f"Error in About tab: {e}")

with tab2:
    try:
        st.markdown("### 🗺️ Map of Top High-Risk Intersections")
        if not intersection_accident_counts.empty:
            map_data = intersection_accident_counts[['latitude', 'longitude', 'accident_frequency']].head(top_n)
            max_freq_map = intersection_accident_counts['accident_frequency'].max()
            scale_factor_map = 10
            map_data['marker_size'] = map_data['accident_frequency'] / max_freq_map * scale_factor_map
            st.map(map_data, size='marker_size', color='accident_frequency', color_scale='plasma')
            st.caption("Higher intensity (size/color) indicates higher accident frequency.")
        else:
            st.warning("No data to display on the map based on current filters.")
    except Exception as e:
        st.error(f"Error in Map tab: {e}")

with tab3:
    try:
        st.markdown("### 🚦 Top Intersections with Highest Accident Frequency")
        if not intersection_accident_counts.empty:
            display_df = intersection_accident_counts[['rounded_location', 'latitude', 'longitude', 'accident_frequency']].head(top_n)
            display_df.columns = ['Location (Lat, Lon)', 'Latitude', 'Longitude', 'Accident Count']
            st.dataframe(display_df)
            st.markdown("### 📊 Accident Frequency by Intersection (Bar Chart)")
            fig_bar = px.bar(display_df, x='Location (Lat, Lon)', y='Accident Count', color='Accident Count', color_continuous_scale='Reds', title=f'Top {top_n} Intersections with Highest Accident Frequency', labels={'Accident Count': 'Accidents'}, height=400)
            fig_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)
        if df_filtered is not None and not df_filtered.empty:
            st.markdown("### 🥧 Accident Severity Breakdown")
            severity_distribution = df_filtered['accident_severity'].map({1: "Fatal", 2: "Serious", 3: "Slight"}).value_counts().reset_index()
            severity_distribution.columns = ['Severity', 'Count']
            fig_pie = px.pie(severity_distribution, names='Severity', values='Count', title='Accident Severity Distribution in Filtered Data', color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_pie, use_container_width=True)
        elif df_filtered is None:
            st.warning("Data filtering failed, cannot display severity breakdown.")
        elif df_filtered.empty:
            st.info("No data available based on the current filters to display severity breakdown.")
    except Exception as e:
        st.error(f"Error in Data & Stats tab: {e}")

with tab4:
    try:
        st.markdown("### 🧠 Insights & Observations")
        if not intersection_accident_counts.empty:
            most_accident_prone = intersection_accident_counts.iloc[0]
            st.markdown(f"- 🚨 The intersection at **{most_accident_prone['Location (Lat, Lon)']}** recorded the **highest number of accidents**: **{most_accident_prone['Accident Count']}**.")
            st.markdown(f"- 📌 The top {top_n} intersections collectively account for **{intersection_accident_counts['accident_frequency'].head(top_n).sum()}** reported accidents in the selected filters.")
        elif df_filtered is not None:
            st.info("No high-risk intersections to highlight based on the current filters.")
        elif df_filtered is None:
            st.warning("Data filtering failed, cannot provide insights.")
        if df_filtered is not None:
            st.markdown(f"- 🧾 Filter applied: **{len(df_filtered)}** accidents matched your criteria.")
        elif df_filtered is None:
            st.warning("Data filtering failed, cannot show filter details.")
    except Exception as e:
        st.error(f"Error in Insights tab: {e}")

with tab5:
    try:
        st.markdown("### 📤 Download Top Intersections Data")
        if not intersection_accident_counts.empty:
            csv_download = intersection_accident_counts[['rounded_location', 'latitude', 'longitude', 'accident_frequency']].head(top_n).to_csv(index=False).encode('utf-8')
            st.download_button(label="Download as CSV", data=csv_download, file_name=f"top_{top_n}_intersections.csv", mime="text/csv")
        else:
            st.warning("No top intersections data available for download based on the current filters.")
    except Exception as e:
        st.error(f"Error in Download tab: {e}")

with tab6:
    accident_severity_prediction_tab(df_merged)

# -------------------------
# ✅ End of App
# -------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and UK Road Safety Data (DfT, 2023)")