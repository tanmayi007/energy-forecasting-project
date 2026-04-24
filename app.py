# -*- coding: utf-8 -*-
"""
Energy Forecasting Dashboard - IMPROVED
Interactive Streamlit frontend for ML energy prediction model
With enhanced color palette, clustering analysis, and visible metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="⚡ Energy Forecast Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with vibrant color palette
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Space Mono', monospace;
        letter-spacing: -0.5px;
        font-weight: 700;
    }
    
    /* Sidebar - Dark theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%) !important;
    }
    
    /* Main background */
    .main {
        background-color: #F8FAFC !important;
    }
    
    /* Enhanced Metric Cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
        padding: 1.5rem !important;
        border-radius: 14px !important;
        border-left: 5px solid #3B82F6 !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08) !important;
    }
    
    [data-testid="stMetric"] > div:first-child {
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
    [data-testid="stMetric"] > div:nth-child(2) {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
    }
    
    [data-testid="stMetric"] > div:nth-child(3) {
        color: #64748B !important;
        font-size: 0.85rem !important;
        margin-top: 0.5rem;
    }
    
    /* Button styling */
    button {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4) !important;
    }
    
    /* Card styling */
    .forecast-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
        padding: 2rem;
        border-radius: 14px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
        border-top: 4px solid #3B82F6;
    }
    
    /* Cluster cards */
    .cluster-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
        border-left: 5px solid;
    }
    
    .cluster-low { border-left-color: #3B82F6; }
    .cluster-moderate { border-left-color: #10B981; }
    .cluster-emerging { border-left-color: #F97316; }
    .cluster-intensive { border-left-color: #EF4444; }
    
    /* Status badges */
    .status-surplus {
        color: #10B981;
        background: #ECFDF5;
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    
    .status-ontrack {
        color: #F97316;
        background: #FFEDD5;
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    
    .status-atrisk {
        color: #EF4444;
        background: #FEE2E2;
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    
    .status-critical {
        color: #991B1B;
        background: #FEE2E2;
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    
    /* Tab styling */
    [data-testid="stTabs"] button {
        color: #64748B !important;
        border-bottom: 2px solid transparent !important;
    }
    
    [data-testid="stTabs"] [aria-selected="true"] button {
        color: #3B82F6 !important;
        border-bottom-color: #3B82F6 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# =====================================================
# LOAD MODEL & DATA (CACHED)
# =====================================================
@st.cache_resource
def load_model_and_data():
    """Load the trained models and preprocessed data"""
    try:
        df = pd.read_excel("enriched_data.xlsx")
        df.columns = df.columns.str.strip()
        df = df.sort_values(["state", "year"]).reset_index(drop=True)
        
        # Feature engineering
        df["renewable_capacity"] = (
            df["solar_capacity"].fillna(0) +
            df["wind_capacity"].fillna(0) +
            df["hydro_capacity"].fillna(0)
        )
        df["supply_gap"] = df["peak_demand(mw)"] - df["renewable_capacity"]
        df["self_sufficiency_ratio"] = df["renewable_capacity"] / (df["peak_demand(mw)"] + 1e-6)
        df["solar_efficiency"] = df["solar_capacity"] / (df["solar_irradiance"] + 1e-6)
        df["lag_1"] = df.groupby("state")["peak_demand(mw)"].shift(1).bfill()
        df["lag_2"] = df.groupby("state")["peak_demand(mw)"].shift(2).bfill()
        
        return df
    except Exception as e:
        st.error(f"⚠️ Error loading data: {e}")
        return None

@st.cache_data
def perform_clustering(df):
    """Perform KMeans clustering on the data"""
    cluster_features = df[[
        "solar_irradiance",
        "wind_speed",
        "solar_capacity",
        "wind_capacity",
        "energy_consumption"
    ]].fillna(0)
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_features)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled)
    
    cluster_labels = {
        0: "Low Infrastructure",
        1: "Moderate Growth",
        2: "High Demand Emerging",
        3: "Energy Intensive Leaders"
    }
    
    return clusters, scaler, kmeans, cluster_labels

# Load data
df = load_model_and_data()

if df is None:
    st.error("Unable to load data. Please ensure 'enriched_data.xlsx' is in the working directory.")
    st.stop()

# Perform clustering
clusters, scaler, kmeans, cluster_labels = perform_clustering(df)
df["cluster"] = clusters
df["cluster_name"] = df["cluster"].map(cluster_labels)

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
st.sidebar.markdown("## ⚡ Navigation")
page = st.sidebar.radio(
    "Select View",
    ["🏠 Dashboard", "🎯 Clustering", "📊 Analytics", "🔮 Forecast", "⚙️ Scenarios", "📈 Rankings"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📌 About
Energy Forecasting dashboard using ML to predict:
- Peak electricity demand
- Renewable energy supply
- Supply-demand gaps
- State clustering & segmentation

**Features:** Interactive forecasts, scenario analysis, real-time rankings
""")

# =====================================================
# PAGE: DASHBOARD
# =====================================================
if page == "🏠 Dashboard":
    st.markdown("# ⚡ Energy Forecast Hub")
    st.markdown("Real-time energy demand and supply analytics for India")
    
    # Get latest data
    latest = df.sort_values("year").groupby("state").last().reset_index()
    
    # Key metrics with actual values
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_demand = latest["peak_demand(mw)"].mean()
        st.metric(
            "📊 Avg Peak Demand",
            f"{avg_demand:,.0f} MW",
            f"across {len(latest)} states"
        )
    
    with col2:
        avg_supply = latest["renewable_capacity"].mean()
        st.metric(
            "♻️ Avg Renewable Supply",
            f"{avg_supply:,.0f} MW",
            "current capacity"
        )
    
    with col3:
        avg_gap = latest["supply_gap"].mean()
        gap_color = "↓" if avg_gap < 0 else "↑"
        st.metric(
            "⚡ Avg Supply Gap",
            f"{avg_gap:,.0f} MW",
            f"{gap_color} {'Surplus' if avg_gap < 0 else 'Deficit'}"
        )
    
    with col4:
        avg_sufficiency = latest["self_sufficiency_ratio"].mean()
        st.metric(
            "🎯 Avg Self-Sufficiency",
            f"{avg_sufficiency:.1%}",
            "renewable ratio"
        )
    
    st.markdown("---")
    
    # Demand vs Supply comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Demand vs Supply Distribution")
        fig_dist = go.Figure()
        
        fig_dist.add_trace(go.Box(
            y=latest["peak_demand(mw)"],
            name="Peak Demand",
            marker_color="#EF4444"
        ))
        fig_dist.add_trace(go.Box(
            y=latest["renewable_capacity"],
            name="Renewable Supply",
            marker_color="#10B981"
        ))
        
        fig_dist.update_layout(
            height=400,
            showlegend=True,
            hovermode='closest',
            template="plotly_white"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        st.markdown("### 🌍 State-wise Supply Gap")
        gap_data = latest.nlargest(10, "supply_gap")[["state", "supply_gap"]].copy()
        
        colors = ["#EF4444" if x > 0 else "#10B981" for x in gap_data["supply_gap"]]
        
        fig_gap = go.Figure(data=[
            go.Bar(
                x=gap_data["state"],
                y=gap_data["supply_gap"],
                marker_color=colors,
                text=gap_data["supply_gap"].round(0),
                textposition="auto"
            )
        ])
        
        fig_gap.update_layout(
            title="Top 10 States by Supply Gap",
            xaxis_title="State",
            yaxis_title="Gap (MW)",
            height=400,
            template="plotly_white",
            hovermode='x'
        )
        st.plotly_chart(fig_gap, use_container_width=True)
    
    st.markdown("---")
    
    # State selector for detailed view
    selected_state = st.selectbox("🔍 Select State for Details", latest["state"].unique())
    
    state_data = latest[latest["state"] == selected_state].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"### 🏛️ {selected_state}")
        st.metric("Peak Demand", f"{state_data['peak_demand(mw)']:,.0f} MW")
        st.metric("Solar Capacity", f"{state_data['solar_capacity']:,.0f} MW")
    
    with col2:
        st.metric("Wind Capacity", f"{state_data['wind_capacity']:,.0f} MW")
        st.metric("Hydro Capacity", f"{state_data['hydro_capacity']:,.0f} MW")
    
    with col3:
        st.metric("Self-Sufficiency", f"{state_data['self_sufficiency_ratio']:.1%}")
        st.markdown(f"**Cluster:** {state_data.get('cluster_name', 'N/A')}")

# =====================================================
# PAGE: CLUSTERING
# =====================================================
elif page == "🎯 Clustering":
    st.markdown("# 🎯 State Clustering & Segmentation")
    st.markdown("Analyze states by infrastructure maturity and energy profiles")
    
    st.markdown("---")
    
    # Get latest cluster assignments
    latest_clusters = df.sort_values("year").groupby("state").last().reset_index()
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    cluster_counts = latest_clusters["cluster_name"].value_counts()
    
    with col1:
        count = len(latest_clusters[latest_clusters["cluster_name"] == "Low Infrastructure"])
        st.metric("🔵 Low Infrastructure", count, "states")
    
    with col2:
        count = len(latest_clusters[latest_clusters["cluster_name"] == "Moderate Growth"])
        st.metric("🟢 Moderate Growth", count, "states")
    
    with col3:
        count = len(latest_clusters[latest_clusters["cluster_name"] == "High Demand Emerging"])
        st.metric("🟠 High Demand", count, "states")
    
    with col4:
        count = len(latest_clusters[latest_clusters["cluster_name"] == "Energy Intensive Leaders"])
        st.metric("🔴 Intensive Leaders", count, "states")
    
    st.markdown("---")
    
    # Cluster detailed breakdown
    st.markdown("## 📋 Cluster Profiles")
    
    # Cluster 0: Low Infrastructure
    with st.expander("🔵 Low Infrastructure - Emerging Markets"):
        cluster_0 = latest_clusters[latest_clusters["cluster_name"] == "Low Infrastructure"]
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**States ({len(cluster_0)}):** {', '.join(cluster_0['state'].unique())}")
            st.metric("Avg Demand", f"{cluster_0['peak_demand(mw)'].mean():,.0f} MW")
            st.metric("Avg Supply", f"{cluster_0['renewable_capacity'].mean():,.0f} MW")
        
        with col2:
            st.metric("Avg Self-Sufficiency", f"{cluster_0['self_sufficiency_ratio'].mean():.1%}")
            st.metric("Avg Consumption", f"{cluster_0['energy_consumption'].mean():,.0f}")
            st.metric("Population Avg", f"{cluster_0['population'].mean()/1e6:.1f}M")
    
    # Cluster 1: Moderate Growth
    with st.expander("🟢 Moderate Growth - Developing Markets"):
        cluster_1 = latest_clusters[latest_clusters["cluster_name"] == "Moderate Growth"]
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**States ({len(cluster_1)}):** {', '.join(cluster_1['state'].unique())}")
            st.metric("Avg Demand", f"{cluster_1['peak_demand(mw)'].mean():,.0f} MW")
            st.metric("Avg Supply", f"{cluster_1['renewable_capacity'].mean():,.0f} MW")
        
        with col2:
            st.metric("Avg Self-Sufficiency", f"{cluster_1['self_sufficiency_ratio'].mean():.1%}")
            st.metric("Avg Consumption", f"{cluster_1['energy_consumption'].mean():,.0f}")
            st.metric("Population Avg", f"{cluster_1['population'].mean()/1e6:.1f}M")
    
    # Cluster 2: High Demand Emerging
    with st.expander("🟠 High Demand Emerging - Growth Markets"):
        cluster_2 = latest_clusters[latest_clusters["cluster_name"] == "High Demand Emerging"]
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**States ({len(cluster_2)}):** {', '.join(cluster_2['state'].unique())}")
            st.metric("Avg Demand", f"{cluster_2['peak_demand(mw)'].mean():,.0f} MW")
            st.metric("Avg Supply", f"{cluster_2['renewable_capacity'].mean():,.0f} MW")
        
        with col2:
            st.metric("Avg Self-Sufficiency", f"{cluster_2['self_sufficiency_ratio'].mean():.1%}")
            st.metric("Avg Consumption", f"{cluster_2['energy_consumption'].mean():,.0f}")
            st.metric("Population Avg", f"{cluster_2['population'].mean()/1e6:.1f}M")
    
    # Cluster 3: Energy Intensive Leaders
    with st.expander("🔴 Energy Intensive Leaders - Developed Markets"):
        cluster_3 = latest_clusters[latest_clusters["cluster_name"] == "Energy Intensive Leaders"]
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**States ({len(cluster_3)}):** {', '.join(cluster_3['state'].unique())}")
            st.metric("Avg Demand", f"{cluster_3['peak_demand(mw)'].mean():,.0f} MW")
            st.metric("Avg Supply", f"{cluster_3['renewable_capacity'].mean():,.0f} MW")
        
        with col2:
            st.metric("Avg Self-Sufficiency", f"{cluster_3['self_sufficiency_ratio'].mean():.1%}")
            st.metric("Avg Consumption", f"{cluster_3['energy_consumption'].mean():,.0f}")
            st.metric("Population Avg", f"{cluster_3['population'].mean()/1e6:.1f}M")
    
    st.markdown("---")
    
    # Cluster visualization
    st.markdown("## 📊 Cluster Analysis Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ☀️ Solar vs Wind Capacity by Cluster")
        
        fig_scatter = px.scatter(
            latest_clusters,
            x="solar_capacity",
            y="wind_capacity",
            color="cluster_name",
            size="peak_demand(mw)",
            hover_name="state",
            color_discrete_map={
                "Low Infrastructure": "#3B82F6",
                "Moderate Growth": "#10B981",
                "High Demand Emerging": "#F97316",
                "Energy Intensive Leaders": "#EF4444"
            },
            title="Solar vs Wind Capacity Distribution",
            labels={"solar_capacity": "Solar Capacity (MW)", "wind_capacity": "Wind Capacity (MW)"}
        )
        
        fig_scatter.update_layout(
            height=400,
            template="plotly_white",
            hovermode='closest'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Self-Sufficiency by Cluster")
        
        fig_box = go.Figure()
        
        for cluster_name in latest_clusters["cluster_name"].unique():
            cluster_data = latest_clusters[latest_clusters["cluster_name"] == cluster_name]
            color_map = {
                "Low Infrastructure": "#3B82F6",
                "Moderate Growth": "#10B981",
                "High Demand Emerging": "#F97316",
                "Energy Intensive Leaders": "#EF4444"
            }
            
            fig_box.add_trace(go.Box(
                y=cluster_data["self_sufficiency_ratio"],
                name=cluster_name,
                marker_color=color_map.get(cluster_name, "#3B82F6")
            ))
        
        fig_box.update_layout(
            height=400,
            title="Self-Sufficiency Distribution by Cluster",
            yaxis_title="Self-Sufficiency Ratio",
            template="plotly_white",
            hovermode='closest'
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    st.markdown("---")
    
    # Feature importance for clustering
    st.markdown("## 🎯 Cluster Characteristics")
    
    cluster_summary = latest_clusters.groupby("cluster_name").agg({
        "peak_demand(mw)": "mean",
        "renewable_capacity": "mean",
        "solar_capacity": "mean",
        "wind_capacity": "mean",
        "hydro_capacity": "mean",
        "energy_consumption": "mean",
        "self_sufficiency_ratio": "mean"
    }).round(0)
    
    st.dataframe(cluster_summary.T, use_container_width=True)

# =====================================================
# PAGE: ANALYTICS
# =====================================================
elif page == "📊 Analytics":
    st.markdown("# 📊 Detailed Analytics")
    
    latest = df.sort_values("year").groupby("state").last().reset_index()
    
    # Correlation analysis
    st.markdown("## 🔗 Feature Correlations")
    
    num_df = df.select_dtypes(include=[np.number])
    corr_matrix = num_df.corr()
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text:.2f}',
        textfont={"size": 8},
        hoverongaps=False
    ))
    
    fig_corr.update_layout(
        height=700,
        title="Correlation Matrix of All Features",
        template="plotly_white"
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("---")
    
    # Top correlations with demand
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⬆️ Positive Correlations with Peak Demand")
        demand_corr = corr_matrix["peak_demand(mw)"].sort_values(ascending=False).head(8)
        
        fig_pos = go.Figure(data=[
            go.Bar(
                x=demand_corr.values,
                y=demand_corr.index,
                orientation='h',
                marker_color="#3B82F6"
            )
        ])
        fig_pos.update_layout(
            height=400,
            xaxis_title="Correlation",
            template="plotly_white"
        )
        st.plotly_chart(fig_pos, use_container_width=True)
    
    with col2:
        st.markdown("### ⬇️ Negative Correlations with Peak Demand")
        demand_corr = corr_matrix["peak_demand(mw)"].sort_values(ascending=True).head(8)
        
        fig_neg = go.Figure(data=[
            go.Bar(
                x=demand_corr.values,
                y=demand_corr.index,
                orientation='h',
                marker_color="#EF4444"
            )
        ])
        fig_neg.update_layout(
            height=400,
            xaxis_title="Correlation",
            template="plotly_white"
        )
        st.plotly_chart(fig_neg, use_container_width=True)
    
    st.markdown("---")
    
    # Time series analysis
    st.markdown("## 📈 Time Series Trends")
    
    time_series = df.groupby("year").agg({
        "peak_demand(mw)": "mean",
        "renewable_capacity": "mean",
        "energy_consumption": "mean"
    }).reset_index()
    
    fig_trend = go.Figure()
    
    fig_trend.add_trace(go.Scatter(
        x=time_series["year"],
        y=time_series["peak_demand(mw)"],
        mode='lines+markers',
        name='Avg Peak Demand',
        line=dict(color="#EF4444", width=3)
    ))
    
    fig_trend.add_trace(go.Scatter(
        x=time_series["year"],
        y=time_series["renewable_capacity"],
        mode='lines+markers',
        name='Avg Renewable Capacity',
        line=dict(color="#10B981", width=3)
    ))
    
    fig_trend.update_layout(
        height=400,
        title="National Average Energy Trends",
        xaxis_title="Year",
        yaxis_title="MW",
        hovermode='x unified',
        template="plotly_white"
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# =====================================================
# PAGE: FORECAST
# =====================================================
elif page == "🔮 Forecast":
    st.markdown("# 🔮 Future Forecast (2025-2030)")
    
    # Scenario controls
    st.markdown("## ⚙️ Forecast Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        solar_growth = st.slider(
            "☀️ Solar Growth Rate",
            min_value=1.0,
            max_value=1.2,
            value=1.04,
            step=0.01,
            help="Annual multiplication factor for solar capacity"
        )
    
    with col2:
        wind_growth = st.slider(
            "💨 Wind Growth Rate",
            min_value=1.0,
            max_value=1.2,
            value=1.035,
            step=0.01
        )
    
    with col3:
        hydro_growth = st.slider(
            "💧 Hydro Growth Rate",
            min_value=1.0,
            max_value=1.2,
            value=1.01,
            step=0.01
        )
    
    # Generate forecast
    future_rows = []
    future_years = range(2025, 2031)
    
    for state in df["state"].unique():
        last = df[df["state"] == state].iloc[-1].copy()
        
        for y in future_years:
            row = last.copy()
            row["year"] = y
            row["energy_consumption"] *= 1.03
            row["population"] *= 1.01
            row["solar_capacity"] *= solar_growth
            row["wind_capacity"] *= wind_growth
            row["hydro_capacity"] *= hydro_growth
            row["lag_1"] = last["peak_demand(mw)"]
            row["lag_2"] = last.get("lag_1", last["peak_demand(mw)"])
            
            future_rows.append(row)
    
    future_df = pd.DataFrame(future_rows)
    
    # Calculate renewable capacity
    future_df["renewable_capacity"] = (
        future_df["solar_capacity"] +
        future_df["wind_capacity"] +
        future_df["hydro_capacity"]
    )
    
    st.markdown("---")
    
    # National forecast visualization
    st.markdown("## 📊 National Forecast")
    
    national_forecast = future_df.groupby("year").agg({
        "peak_demand(mw)": "mean",
        "renewable_capacity": "mean",
        "energy_consumption": "mean"
    }).reset_index()
    
    # Add historical data
    historical = df.groupby("year").agg({
        "peak_demand(mw)": "mean",
        "renewable_capacity": "mean"
    }).reset_index()
    
    fig_forecast = go.Figure()
    
    fig_forecast.add_trace(go.Scatter(
        x=historical["year"],
        y=historical["peak_demand(mw)"],
        mode='lines+markers',
        name='Historical Demand',
        line=dict(color="#EF4444", width=2, dash="dash")
    ))
    
    fig_forecast.add_trace(go.Scatter(
        x=national_forecast["year"],
        y=national_forecast["peak_demand(mw)"],
        mode='lines+markers',
        name='Forecast Demand',
        line=dict(color="#EF4444", width=3)
    ))
    
    fig_forecast.add_trace(go.Scatter(
        x=historical["year"],
        y=historical["renewable_capacity"],
        mode='lines+markers',
        name='Historical Supply',
        line=dict(color="#10B981", width=2, dash="dash")
    ))
    
    fig_forecast.add_trace(go.Scatter(
        x=national_forecast["year"],
        y=national_forecast["renewable_capacity"],
        mode='lines+markers',
        name='Forecast Supply',
        line=dict(color="#10B981", width=3)
    ))
    
    fig_forecast.update_layout(
        height=500,
        title="National Energy Forecast (with Growth Multipliers)",
        xaxis_title="Year",
        yaxis_title="MW",
        hovermode='x unified',
        template="plotly_white"
    )
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    st.markdown("---")
    
    # State-specific forecast
    st.markdown("## 🔍 State-wise Forecast")
    
    state_options = sorted(future_df["state"].unique())
    selected_states = st.multiselect(
        "Select States to Compare",
        state_options,
        default=state_options[:3]
    )
    
    if selected_states:
        state_forecast = future_df[future_df["state"].isin(selected_states)].groupby(
            ["year", "state"]
        ).agg({
            "peak_demand(mw)": "mean",
            "renewable_capacity": "mean"
        }).reset_index()
        
        state_forecast["supply_gap"] = (
            state_forecast["peak_demand(mw)"] - state_forecast["renewable_capacity"]
        )
        
        fig_state = px.line(
            state_forecast,
            x="year",
            y="supply_gap",
            color="state",
            markers=True,
            title="Supply Gap Forecast by State",
            labels={"supply_gap": "Gap (MW)", "year": "Year"}
        )
        
        fig_state.update_layout(
            height=400,
            hovermode='x unified',
            template="plotly_white"
        )
        st.plotly_chart(fig_state, use_container_width=True)

# =====================================================
# PAGE: SCENARIOS
# =====================================================
elif page == "⚙️ Scenarios":
    st.markdown("# ⚙️ Scenario Analysis")
    st.markdown("Simulate different policy and infrastructure scenarios")
    
    st.markdown("---")
    
    # Scenario definition
    st.markdown("## 🎯 Create Custom Scenario")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        scenario_name = st.text_input("Scenario Name", "Custom Scenario")
    
    with col2:
        scenario_year = st.selectbox("Target Year", range(2025, 2031), index=3)
    
    with col3:
        st.write("")  # spacing
    
    st.markdown("### National Multipliers")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        solar_mult = st.slider(
            "Solar Capacity Multiplier",
            min_value=0.8,
            max_value=2.0,
            value=1.0,
            step=0.05,
            key="scenario_solar"
        )
    
    with col2:
        wind_mult = st.slider(
            "Wind Capacity Multiplier",
            min_value=0.8,
            max_value=2.0,
            value=1.0,
            step=0.05,
            key="scenario_wind"
        )
    
    with col3:
        hydro_mult = st.slider(
            "Hydro Capacity Multiplier",
            min_value=0.8,
            max_value=2.0,
            value=1.0,
            step=0.05,
            key="scenario_hydro"
        )
    
    st.markdown("---")
    
    # State-specific overrides
    st.markdown("### 🔧 State-Specific Adjustments")
    
    states_to_adjust = st.multiselect(
        "Select States to Adjust",
        sorted(df["state"].unique()),
        default=[]
    )
    
    state_multipliers = {}
    
    if states_to_adjust:
        for state in states_to_adjust:
            with st.expander(f"⚡ {state}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    s_mult = st.slider(
                        "Solar",
                        min_value=0.5,
                        max_value=2.0,
                        value=1.0,
                        step=0.05,
                        key=f"{state}_solar"
                    )
                
                with col2:
                    w_mult = st.slider(
                        "Wind",
                        min_value=0.5,
                        max_value=2.0,
                        value=1.0,
                        step=0.05,
                        key=f"{state}_wind"
                    )
                
                with col3:
                    h_mult = st.slider(
                        "Hydro",
                        min_value=0.5,
                        max_value=2.0,
                        value=1.0,
                        step=0.05,
                        key=f"{state}_hydro"
                    )
                
                state_multipliers[state] = {
                    "solar": s_mult,
                    "wind": w_mult,
                    "hydro": h_mult
                }
    
    st.markdown("---")
    
    # Run scenario
    if st.button("🚀 Run Scenario Analysis", key="run_scenario"):
        st.markdown(f"## 📊 Results for {scenario_name}")
        
        # Apply scenario to current latest data
        latest = df.sort_values("year").groupby("state").last().reset_index()
        scenario_data = latest.copy()
        
        # National multipliers
        scenario_data["solar_capacity"] *= solar_mult
        scenario_data["wind_capacity"] *= wind_mult
        scenario_data["hydro_capacity"] *= hydro_mult
        
        # State-specific
        for state, mults in state_multipliers.items():
            mask = scenario_data["state"] == state
            if "solar" in mults:
                scenario_data.loc[mask, "solar_capacity"] *= mults["solar"]
            if "wind" in mults:
                scenario_data.loc[mask, "wind_capacity"] *= mults["wind"]
            if "hydro" in mults:
                scenario_data.loc[mask, "hydro_capacity"] *= mults["hydro"]
        
        scenario_data["renewable_capacity"] = (
            scenario_data["solar_capacity"] +
            scenario_data["wind_capacity"] +
            scenario_data["hydro_capacity"]
        )
        scenario_data["supply_gap"] = (
            scenario_data["peak_demand(mw)"] - scenario_data["renewable_capacity"]
        )
        scenario_data["self_sufficiency_ratio"] = (
            scenario_data["renewable_capacity"] / (scenario_data["peak_demand(mw)"] + 1e-6)
        )
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Avg Supply Gap",
                f"{scenario_data['supply_gap'].mean():,.0f} MW",
                f"{(scenario_data['supply_gap'].mean() / latest['supply_gap'].mean() - 1) * 100:.1f}% vs current"
            )
        
        with col2:
            st.metric(
                "Avg Self-Sufficiency",
                f"{scenario_data['self_sufficiency_ratio'].mean():.1%}",
                f"{(scenario_data['self_sufficiency_ratio'].mean() / latest['self_sufficiency_ratio'].mean() - 1) * 100:.1f}% improvement"
            )
        
        with col3:
            surplus_states = len(scenario_data[scenario_data["supply_gap"] < 0])
            st.metric(
                "States with Surplus",
                surplus_states,
                f"out of {len(scenario_data)}"
            )
        
        with col4:
            critical_states = len(scenario_data[scenario_data["self_sufficiency_ratio"] < 0.2])
            st.metric(
                "Critical States",
                critical_states,
                "self-sufficiency < 20%"
            )
        
        st.markdown("---")
        
        # Comparison chart
        comparison_df = pd.DataFrame({
            "Scenario": ["Current"] * len(latest) + ["After Scenario"] * len(scenario_data),
            "State": list(latest["state"]) + list(scenario_data["state"]),
            "Self-Sufficiency": list(latest["self_sufficiency_ratio"]) + list(scenario_data["self_sufficiency_ratio"])
        })
        
        fig_comparison = px.bar(
            comparison_df,
            x="State",
            y="Self-Sufficiency",
            color="Scenario",
            barmode="group",
            title="Self-Sufficiency: Before vs After Scenario",
            height=400
        )
        
        fig_comparison.update_layout(
            template="plotly_white",
            hovermode='x unified'
        )
        st.plotly_chart(fig_comparison, use_container_width=True)

# =====================================================
# PAGE: RANKINGS
# =====================================================
elif page == "📈 Rankings":
    st.markdown("# 📈 State Rankings & Status")
    
    latest = df.sort_values("year").groupby("state").last().reset_index()
    
    # Add status classification
    def classify_status(x):
        if x > 0.6:
            return "🟢 Surplus"
        elif x > 0.4:
            return "🟡 On Track"
        elif x > 0.2:
            return "🟠 At Risk"
        return "🔴 Critical"
    
    latest["status"] = latest["self_sufficiency_ratio"].apply(classify_status)
    
    # Ranking tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌱 Self-Sufficiency",
        "⚡ Peak Demand",
        "☀️ Solar Efficiency",
        "⚠️ Supply Gap"
    ])
    
    with tab1:
        st.markdown("## Renewable Energy Self-Sufficiency Ranking")
        ranking = latest[["state", "self_sufficiency_ratio", "status"]].sort_values(
            "self_sufficiency_ratio", ascending=False
        ).reset_index(drop=True)
        ranking["Rank"] = range(1, len(ranking) + 1)
        
        st.dataframe(
            ranking[["Rank", "state", "self_sufficiency_ratio", "status"]].rename(
                columns={
                    "state": "State",
                    "self_sufficiency_ratio": "Ratio",
                    "status": "Status"
                }
            ),
            use_container_width=True,
            hide_index=True
        )
        
        fig = px.bar(
            ranking,
            x="state",
            y="self_sufficiency_ratio",
            color="self_sufficiency_ratio",
            color_continuous_scale="RdYlGn",
            title="Self-Sufficiency Ratio by State",
            labels={"self_sufficiency_ratio": "Ratio"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("## Peak Demand Ranking")
        ranking = latest[["state", "peak_demand(mw)"]].sort_values(
            "peak_demand(mw)", ascending=False
        ).reset_index(drop=True)
        ranking["Rank"] = range(1, len(ranking) + 1)
        
        st.dataframe(
            ranking[["Rank", "state", "peak_demand(mw)"]].rename(
                columns={"state": "State", "peak_demand(mw)": "Demand (MW)"}
            ),
            use_container_width=True,
            hide_index=True
        )
        
        fig = px.bar(
            ranking,
            x="state",
            y="peak_demand(mw)",
            color="peak_demand(mw)",
            color_continuous_scale="Reds",
            title="Peak Demand by State",
            labels={"peak_demand(mw)": "Demand (MW)"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("## Solar Efficiency Ranking")
        ranking = latest[["state", "solar_efficiency"]].sort_values(
            "solar_efficiency", ascending=False
        ).reset_index(drop=True)
        ranking = ranking[ranking["solar_efficiency"].notna()]
        ranking["Rank"] = range(1, len(ranking) + 1)
        
        st.dataframe(
            ranking[["Rank", "state", "solar_efficiency"]].rename(
                columns={"state": "State", "solar_efficiency": "Efficiency"}
            ),
            use_container_width=True,
            hide_index=True
        )
        
        fig = px.bar(
            ranking,
            x="state",
            y="solar_efficiency",
            color="solar_efficiency",
            color_continuous_scale="Oranges",
            title="Solar Efficiency by State"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("## Supply Gap Ranking")
        ranking = latest[["state", "supply_gap"]].sort_values(
            "supply_gap", ascending=False
        ).reset_index(drop=True)
        ranking["Rank"] = range(1, len(ranking) + 1)
        
        st.dataframe(
            ranking[["Rank", "state", "supply_gap"]].rename(
                columns={"state": "State", "supply_gap": "Gap (MW)"}
            ),
            use_container_width=True,
            hide_index=True
        )
        
        colors = ["#EF4444" if x > 0 else "#10B981" for x in ranking["supply_gap"]]
        
        fig = go.Figure(data=[
            go.Bar(
                x=ranking["state"],
                y=ranking["supply_gap"],
                marker_color=colors,
                text=ranking["supply_gap"].round(0),
                textposition="auto"
            )
        ])
        
        fig.update_layout(
            title="Supply Gap by State (Positive = Deficit)",
            xaxis_title="State",
            yaxis_title="Gap (MW)",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; margin-top: 2rem;">
    <p>⚡ Energy Forecasting Dashboard | ML-Powered Analytics</p>
    <p style="font-size: 0.85rem;">With Clustering, Forecasting & Scenario Analysis | Updated 2024</p>
</div>
""", unsafe_allow_html=True)