import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium

# ——— Load Data ———
@st.cache_data

def load_data():
    df = pd.read_csv("datasets/dft-road-casualty-statistics-collision-2023.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M').dt.hour
    return df

df = load_data()

# ——— Sidebar Filters ———
st.sidebar.header("🔍 Filter Data")
city = st.sidebar.selectbox("City", ["All"] + sorted(df['City'].unique().tolist()))
date_range = st.sidebar.date_input("Date Range", [df['Date'].min(), df['Date'].max()])
severity = st.sidebar.multiselect("Severity", options=df['Severity'].unique(), default=df['Severity'].unique())

filtered_df = df.copy()
if city != "All":
    filtered_df = filtered_df[filtered_df['City'] == city]
filtered_df = filtered_df[(filtered_df['Date'] >= pd.to_datetime(date_range[0])) & (filtered_df['Date'] <= pd.to_datetime(date_range[1]))]
filtered_df = filtered_df[filtered_df['Severity'].isin(severity)]

# ——— Tabs ———
tabs = st.tabs(["🧭 Overview", "🗺️ Map", "📊 Charts", "🧠 Insights", "📥 Export", "🤖 ML Predictions"])

# ——— Overview Tab ———
with tabs[0]:
    st.title("🚦 Road Safety Analysis Dashboard")
    st.markdown("This dashboard provides insights into road accident data for better decision-making and safety improvements.")
    st.metric("Total Accidents", len(filtered_df))
    st.metric("Average Severity", round(filtered_df['Severity'].mean(), 2))
    st.metric("Cities Affected", filtered_df['City'].nunique())

# ——— Map Tab ———
with tabs[1]:
    st.subheader("📍 Accident Map")
    m = folium.Map(location=[filtered_df['Latitude'].mean(), filtered_df['Longitude'].mean()], zoom_start=11)
    for _, row in filtered_df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=4,
            popup=f"{row['City']} | Severity: {row['Severity']}",
            color='red',
            fill=True,
            fill_color='red'
        ).add_to(m)
    st_folium(m, width=700, height=500)

# ——— Charts Tab ———
with tabs[2]:
    st.subheader("📊 Accident Charts")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.pie(filtered_df, names='Severity', title='Severity Distribution')
        st.plotly_chart(fig1)

    with col2:
        fig2 = px.bar(filtered_df['City'].value_counts().reset_index(),
                      x='index', y='City',
                      labels={'index': 'City', 'City': 'Accident Count'},
                      title='Accidents per City')
        st.plotly_chart(fig2)

    fig3 = px.histogram(filtered_df, x='Hour', nbins=24, title='Accidents by Hour')
    st.plotly_chart(fig3, use_container_width=True)

# ——— Insights Tab ———
with tabs[3]:
    st.subheader("🧠 Key Insights")
    top_cities = filtered_df['City'].value_counts().head(5)
    st.write("**Top 5 Cities with Most Accidents:**")
    st.dataframe(top_cities)

    severe_cases = filtered_df[filtered_df['Severity'] >= 3]
    st.write("**High Severity Cases (% of Total):**", f"{len(severe_cases)/len(filtered_df)*100:.2f}%")

    peak_hour = filtered_df['Hour'].value_counts().idxmax()
    st.write(f"**Peak Hour for Accidents:** {peak_hour}:00")

# ——— Export Tab ———
with tabs[4]:
    st.subheader("📥 Export Data")
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name='filtered_accidents.csv',
        mime='text/csv'
    )

# ——— ML Predictions (Optional) ———
with tabs[5]:
    st.subheader("🤖 ML Predictions")
    st.info("ML Model Integration Coming Soon! You can integrate an accident severity predictor here using scikit-learn or TensorFlow.")
