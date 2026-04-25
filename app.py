# -*- coding: utf-8 -*-
"""
Energy Forecasting Dashboard v3.1 - FIXED
Using native Streamlit st.metric with proper styling
"""

import streamlit as st
import pandas as pd
import numpy as np
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

# =====================================================
# LOAD & CLUSTER DATA
# =====================================================
@st.cache_resource
def load_model_and_data():
    try:
        df = pd.read_excel("enriched_data.xlsx")
        df.columns = df.columns.str.strip()
        df = df.sort_values(["state", "year"]).reset_index(drop=True)
        
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

df = load_model_and_data()
if df is None:
    st.error("Unable to load data. Ensure 'enriched_data.xlsx' exists.")
    st.stop()

clusters, scaler, kmeans, cluster_labels = perform_clustering(df)
df["cluster"] = clusters
df["cluster_name"] = df["cluster"].map(cluster_labels)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.markdown("## ⚡ Navigation")
page = st.sidebar.radio(
    "Select View",
    ["🏠 Dashboard", "🎯 Clustering", "📊 Analytics", "🔮 Forecast", "⚙️ Scenarios", "📈 Rankings"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📌 About
ML-powered energy forecasting for India with:
- Peak demand prediction
- Renewable supply analysis
- State clustering
- Scenario planning
""")

# =====================================================
# PAGE: DASHBOARD
# =====================================================
if page == "🏠 Dashboard":
    st.markdown("# ⚡ Energy Forecast Hub")
    st.markdown("Real-time energy demand and supply analytics for India")
    st.markdown("---")
    
    latest = df.sort_values("year").groupby("state").last().reset_index()
    
    # KPI Cards using native st.metric
    col1, col2, col3, col4 = st.columns(4)
    
    avg_demand = latest["peak_demand(mw)"].mean()
    avg_supply = latest["renewable_capacity"].mean()
    avg_gap = latest["supply_gap"].mean()
    avg_sufficiency = latest["self_sufficiency_ratio"].mean()
    
    with col1:
        st.metric(
            "📊 Peak Demand",
            f"{avg_demand:,.0f}",
            f"MW across {len(latest)} states"
        )
    
    with col2:
        st.metric(
            "♻️ Renewable Supply",
            f"{avg_supply:,.0f}",
            "MW current capacity"
        )
    
    with col3:
        st.metric(
            "⚡ Supply Gap",
            f"{avg_gap:,.0f}",
            "MW Deficit" if avg_gap > 0 else "MW Surplus"
        )
    
    with col4:
        st.metric(
            "🎯 Self-Sufficiency",
            f"{avg_sufficiency:.1%}",
            "Renewable ratio"
        )
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Demand vs Supply Distribution")
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Box(y=latest["peak_demand(mw)"], name="Peak Demand", marker_color="#EF4444"))
        fig_dist.add_trace(go.Box(y=latest["renewable_capacity"], name="Renewable Supply", marker_color="#10B981"))
        fig_dist.update_layout(height=400, template="plotly_white", hovermode='closest')
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        st.markdown("### 🌍 Top 10 States by Supply Gap")
        gap_data = latest.nlargest(10, "supply_gap")[["state", "supply_gap"]].copy()
        colors = ["#EF4444" if x > 0 else "#10B981" for x in gap_data["supply_gap"]]
        fig_gap = go.Figure(data=[go.Bar(x=gap_data["state"], y=gap_data["supply_gap"], marker_color=colors, text=gap_data["supply_gap"].round(0), textposition="auto")])
        fig_gap.update_layout(title="Supply Gap (MW)", height=400, template="plotly_white", hovermode='x')
        st.plotly_chart(fig_gap, use_container_width=True)
    
    st.markdown("---")
    
    # State Details
    st.markdown("### 🔍 Select State for Details")
    selected_state = st.selectbox("Choose a state:", latest["state"].unique(), key="dashboard_state")
    state_data = latest[latest["state"] == selected_state].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"#### 🏛️ {selected_state}")
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
    
    latest_clusters = df.sort_values("year").groupby("state").last().reset_index()
    
    # Cluster counts
    col1, col2, col3, col4 = st.columns(4)
    
    count_low = len(latest_clusters[latest_clusters["cluster_name"] == "Low Infrastructure"])
    count_mod = len(latest_clusters[latest_clusters["cluster_name"] == "Moderate Growth"])
    count_high = len(latest_clusters[latest_clusters["cluster_name"] == "High Demand Emerging"])
    count_intensive = len(latest_clusters[latest_clusters["cluster_name"] == "Energy Intensive Leaders"])
    
    with col1:
        st.metric("🔵 Low Infrastructure", count_low, "Emerging markets")
    with col2:
        st.metric("🟢 Moderate Growth", count_mod, "Developing markets")
    with col3:
        st.metric("🟠 High Demand", count_high, "Growth markets")
    with col4:
        st.metric("🔴 Intensive Leaders", count_intensive, "Developed markets")
    
    st.markdown("---")
    st.markdown("## 📋 Cluster Profiles & Member States")
    
    # Cluster 0
    with st.expander("🔵 Low Infrastructure - Emerging Markets", expanded=False):
        cluster_0 = latest_clusters[latest_clusters["cluster_name"] == "Low Infrastructure"]
        
        st.markdown(f"### States in this cluster ({len(cluster_0)})")
        states_0 = sorted(cluster_0['state'].unique())
        st.markdown(f"**{', '.join(states_0)}**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Demand", f"{cluster_0['peak_demand(mw)'].mean():,.0f} MW")
            st.metric("Avg Supply", f"{cluster_0['renewable_capacity'].mean():,.0f} MW")
        with col2:
            st.metric("Avg Self-Sufficiency", f"{cluster_0['self_sufficiency_ratio'].mean():.1%}")
            st.metric("Avg Consumption", f"{cluster_0['energy_consumption'].mean():,.0f}")
        
        # Show detailed table for this cluster
        st.markdown("#### Detailed Metrics")
        cluster_0_display = cluster_0[["state", "peak_demand(mw)", "renewable_capacity", "self_sufficiency_ratio"]].copy()
        cluster_0_display.columns = ["State", "Peak Demand (MW)", "Renewable Supply (MW)", "Self-Sufficiency"]
        cluster_0_display["Self-Sufficiency"] = cluster_0_display["Self-Sufficiency"].apply(lambda x: f"{x:.1%}")
        st.dataframe(cluster_0_display, use_container_width=True, hide_index=True)
    
    # Cluster 1
    with st.expander("🟢 Moderate Growth - Developing Markets", expanded=False):
        cluster_1 = latest_clusters[latest_clusters["cluster_name"] == "Moderate Growth"]
        
        st.markdown(f"### States in this cluster ({len(cluster_1)})")
        states_1 = sorted(cluster_1['state'].unique())
        st.markdown(f"**{', '.join(states_1)}**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Demand", f"{cluster_1['peak_demand(mw)'].mean():,.0f} MW")
            st.metric("Avg Supply", f"{cluster_1['renewable_capacity'].mean():,.0f} MW")
        with col2:
            st.metric("Avg Self-Sufficiency", f"{cluster_1['self_sufficiency_ratio'].mean():.1%}")
            st.metric("Avg Consumption", f"{cluster_1['energy_consumption'].mean():,.0f}")
        
        # Show detailed table for this cluster
        st.markdown("#### Detailed Metrics")
        cluster_1_display = cluster_1[["state", "peak_demand(mw)", "renewable_capacity", "self_sufficiency_ratio"]].copy()
        cluster_1_display.columns = ["State", "Peak Demand (MW)", "Renewable Supply (MW)", "Self-Sufficiency"]
        cluster_1_display["Self-Sufficiency"] = cluster_1_display["Self-Sufficiency"].apply(lambda x: f"{x:.1%}")
        st.dataframe(cluster_1_display, use_container_width=True, hide_index=True)
    
    # Cluster 2
    with st.expander("🟠 High Demand Emerging - Growth Markets", expanded=False):
        cluster_2 = latest_clusters[latest_clusters["cluster_name"] == "High Demand Emerging"]
        
        st.markdown(f"### States in this cluster ({len(cluster_2)})")
        states_2 = sorted(cluster_2['state'].unique())
        st.markdown(f"**{', '.join(states_2)}**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Demand", f"{cluster_2['peak_demand(mw)'].mean():,.0f} MW")
            st.metric("Avg Supply", f"{cluster_2['renewable_capacity'].mean():,.0f} MW")
        with col2:
            st.metric("Avg Self-Sufficiency", f"{cluster_2['self_sufficiency_ratio'].mean():.1%}")
            st.metric("Avg Consumption", f"{cluster_2['energy_consumption'].mean():,.0f}")
        
        # Show detailed table for this cluster
        st.markdown("#### Detailed Metrics")
        cluster_2_display = cluster_2[["state", "peak_demand(mw)", "renewable_capacity", "self_sufficiency_ratio"]].copy()
        cluster_2_display.columns = ["State", "Peak Demand (MW)", "Renewable Supply (MW)", "Self-Sufficiency"]
        cluster_2_display["Self-Sufficiency"] = cluster_2_display["Self-Sufficiency"].apply(lambda x: f"{x:.1%}")
        st.dataframe(cluster_2_display, use_container_width=True, hide_index=True)
    
    # Cluster 3
    with st.expander("🔴 Energy Intensive Leaders - Developed Markets", expanded=False):
        cluster_3 = latest_clusters[latest_clusters["cluster_name"] == "Energy Intensive Leaders"]
        
        st.markdown(f"### States in this cluster ({len(cluster_3)})")
        states_3 = sorted(cluster_3['state'].unique())
        st.markdown(f"**{', '.join(states_3)}**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Demand", f"{cluster_3['peak_demand(mw)'].mean():,.0f} MW")
            st.metric("Avg Supply", f"{cluster_3['renewable_capacity'].mean():,.0f} MW")
        with col2:
            st.metric("Avg Self-Sufficiency", f"{cluster_3['self_sufficiency_ratio'].mean():.1%}")
            st.metric("Avg Consumption", f"{cluster_3['energy_consumption'].mean():,.0f}")
        
        # Show detailed table for this cluster
        st.markdown("#### Detailed Metrics")
        cluster_3_display = cluster_3[["state", "peak_demand(mw)", "renewable_capacity", "self_sufficiency_ratio"]].copy()
        cluster_3_display.columns = ["State", "Peak Demand (MW)", "Renewable Supply (MW)", "Self-Sufficiency"]
        cluster_3_display["Self-Sufficiency"] = cluster_3_display["Self-Sufficiency"].apply(lambda x: f"{x:.1%}")
        st.dataframe(cluster_3_display, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("## 📊 Cluster Analysis Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ☀️ Solar vs Wind Capacity")
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
            }
        )
        fig_scatter.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Self-Sufficiency by Cluster")
        fig_box = go.Figure()
        for cluster_name in sorted(latest_clusters["cluster_name"].unique()):
            cluster_data = latest_clusters[latest_clusters["cluster_name"] == cluster_name]
            color_map = {"Low Infrastructure": "#3B82F6", "Moderate Growth": "#10B981", "High Demand Emerging": "#F97316", "Energy Intensive Leaders": "#EF4444"}
            fig_box.add_trace(go.Box(y=cluster_data["self_sufficiency_ratio"], name=cluster_name, marker_color=color_map.get(cluster_name, "#3B82F6")))
        fig_box.update_layout(height=400, title="Self-Sufficiency Distribution", template="plotly_white")
        st.plotly_chart(fig_box, use_container_width=True)

# =====================================================
# PAGE: ANALYTICS
# =====================================================
elif page == "📊 Analytics":
    st.markdown("# 📊 Detailed Analytics")
    
    num_df = df.select_dtypes(include=[np.number])
    corr_matrix = num_df.corr()
    
    st.markdown("## 🔗 Feature Correlations")
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0
    ))
    fig_corr.update_layout(height=700, template="plotly_white")
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⬆️ Positive Correlations with Peak Demand")
        demand_corr = corr_matrix["peak_demand(mw)"].sort_values(ascending=False).head(8)
        fig_pos = go.Figure(data=[go.Bar(x=demand_corr.values, y=demand_corr.index, orientation='h', marker_color="#3B82F6")])
        fig_pos.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_pos, use_container_width=True)
    
    with col2:
        st.markdown("### ⬇️ Negative Correlations with Peak Demand")
        demand_corr = corr_matrix["peak_demand(mw)"].sort_values(ascending=True).head(8)
        fig_neg = go.Figure(data=[go.Bar(x=demand_corr.values, y=demand_corr.index, orientation='h', marker_color="#EF4444")])
        fig_neg.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_neg, use_container_width=True)
    
    st.markdown("---")
    st.markdown("## 📈 Time Series Trends")
    
    time_series = df.groupby("year").agg({
        "peak_demand(mw)": "mean",
        "renewable_capacity": "mean"
    }).reset_index()
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=time_series["year"], y=time_series["peak_demand(mw)"], mode='lines+markers', name='Avg Peak Demand', line=dict(color="#EF4444", width=3)))
    fig_trend.add_trace(go.Scatter(x=time_series["year"], y=time_series["renewable_capacity"], mode='lines+markers', name='Avg Renewable Capacity', line=dict(color="#10B981", width=3)))
    fig_trend.update_layout(height=400, title="National Average Energy Trends", template="plotly_white")
    st.plotly_chart(fig_trend, use_container_width=True)

# =====================================================
# PAGE: FORECAST
# =====================================================
elif page == "🔮 Forecast":
    st.markdown("# 🔮 Future Forecast (2025-2030)")
    st.markdown("---")
    
    st.markdown("## ⚙️ Forecast Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        solar_growth = st.slider("☀️ Solar Growth Rate", 1.0, 1.2, 1.04, 0.01)
    with col2:
        wind_growth = st.slider("💨 Wind Growth Rate", 1.0, 1.2, 1.035, 0.01)
    with col3:
        hydro_growth = st.slider("💧 Hydro Growth Rate", 1.0, 1.2, 1.01, 0.01)
    
    # Generate forecast
    future_rows = []
    for state in df["state"].unique():
        last = df[df["state"] == state].iloc[-1].copy()
        for y in range(2025, 2031):
            row = last.copy()
            row["year"] = y
            row["energy_consumption"] *= 1.03
            row["population"] *= 1.01
            row["solar_capacity"] *= solar_growth
            row["wind_capacity"] *= wind_growth
            row["hydro_capacity"] *= hydro_growth
            future_rows.append(row)
    
    future_df = pd.DataFrame(future_rows)
    future_df["renewable_capacity"] = future_df["solar_capacity"] + future_df["wind_capacity"] + future_df["hydro_capacity"]
    
    st.markdown("---")
    st.markdown("## 📊 National Forecast")
    
    national_forecast = future_df.groupby("year").agg({"peak_demand(mw)": "mean", "renewable_capacity": "mean"}).reset_index()
    historical = df.groupby("year").agg({"peak_demand(mw)": "mean", "renewable_capacity": "mean"}).reset_index()
    
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=historical["year"], y=historical["peak_demand(mw)"], mode='lines+markers', name='Historical Demand', line=dict(color="#EF4444", width=2, dash="dash")))
    fig_forecast.add_trace(go.Scatter(x=national_forecast["year"], y=national_forecast["peak_demand(mw)"], mode='lines+markers', name='Forecast Demand', line=dict(color="#EF4444", width=3)))
    fig_forecast.add_trace(go.Scatter(x=historical["year"], y=historical["renewable_capacity"], mode='lines+markers', name='Historical Supply', line=dict(color="#10B981", width=2, dash="dash")))
    fig_forecast.add_trace(go.Scatter(x=national_forecast["year"], y=national_forecast["renewable_capacity"], mode='lines+markers', name='Forecast Supply', line=dict(color="#10B981", width=3)))
    fig_forecast.update_layout(height=500, title="National Energy Forecast", template="plotly_white")
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    st.markdown("---")
    st.markdown("## 🔍 State-Specific Forecast")
    
    state_options = sorted(future_df["state"].unique())
    selected_states = st.multiselect(
        "Select states to view individual forecasts:",
        state_options,
        default=state_options[:3] if len(state_options) >= 3 else state_options,
        key="forecast_states"
    )
    
    if selected_states:
        # Forecast for selected states
        state_forecast = future_df[future_df["state"].isin(selected_states)].groupby(
            ["year", "state"]
        ).agg({
            "peak_demand(mw)": "mean",
            "renewable_capacity": "mean"
        }).reset_index()
        
        state_forecast["supply_gap"] = (
            state_forecast["peak_demand(mw)"] - state_forecast["renewable_capacity"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Demand Forecast")
            fig_demand = px.line(
                state_forecast,
                x="year",
                y="peak_demand(mw)",
                color="state",
                markers=True,
                title="Peak Demand by State (2025-2030)",
                labels={"peak_demand(mw)": "Peak Demand (MW)", "year": "Year"}
            )
            fig_demand.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig_demand, use_container_width=True)
        
        with col2:
            st.markdown("### Supply Forecast")
            fig_supply = px.line(
                state_forecast,
                x="year",
                y="renewable_capacity",
                color="state",
                markers=True,
                title="Renewable Supply by State (2025-2030)",
                labels={"renewable_capacity": "Renewable Capacity (MW)", "year": "Year"}
            )
            fig_supply.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig_supply, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### Supply Gap Forecast")
        fig_gap = px.line(
            state_forecast,
            x="year",
            y="supply_gap",
            color="state",
            markers=True,
            title="Supply Gap by State (2025-2030) - Positive = Deficit",
            labels={"supply_gap": "Gap (MW)", "year": "Year"}
        )
        fig_gap.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_gap, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 📊 Forecast Table for Selected States")
        forecast_table = state_forecast.pivot_table(
            index="state",
            columns="year",
            values="supply_gap",
            aggfunc="first"
        ).round(0)
        forecast_table.columns = [f"Year {int(col)}" for col in forecast_table.columns]
        st.dataframe(forecast_table, use_container_width=True)

# =====================================================
# PAGE: SCENARIOS
# =====================================================
elif page == "⚙️ Scenarios":
    st.markdown("# ⚙️ Scenario Analysis")
    st.markdown("Simulate different policy and infrastructure scenarios")
    st.markdown("---")
    
    st.markdown("## 🎯 Create Custom Scenario")
    scenario_name = st.text_input("Scenario Name", "Custom Scenario")
    
    # National multipliers
    st.markdown("### National Multipliers")
    col1, col2, col3 = st.columns(3)
    with col1:
        solar_mult = st.slider("☀️ Solar Multiplier", 0.8, 2.0, 1.0, 0.05, key="scenario_solar")
    with col2:
        wind_mult = st.slider("💨 Wind Multiplier", 0.8, 2.0, 1.0, 0.05, key="scenario_wind")
    with col3:
        hydro_mult = st.slider("💧 Hydro Multiplier", 0.8, 2.0, 1.0, 0.05, key="scenario_hydro")
    
    st.markdown("---")
    
    # State-specific adjustments
    st.markdown("### 🔧 State-Specific Adjustments (Optional)")
    latest = df.sort_values("year").groupby("state").last().reset_index()
    
    states_to_adjust = st.multiselect(
        "Select states to apply custom multipliers:",
        sorted(latest["state"].unique()),
        key="states_selector"
    )
    
    state_multipliers = {}
    if states_to_adjust:
        st.markdown("#### Configure each state:")
        for state in states_to_adjust:
            with st.expander(f"⚡ {state}", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    s_mult = st.slider(f"☀️ Solar", 0.5, 2.0, 1.0, 0.05, key=f"{state}_solar")
                with col2:
                    w_mult = st.slider(f"💨 Wind", 0.5, 2.0, 1.0, 0.05, key=f"{state}_wind")
                with col3:
                    h_mult = st.slider(f"💧 Hydro", 0.5, 2.0, 1.0, 0.05, key=f"{state}_hydro")
                
                state_multipliers[state] = {
                    "solar": s_mult,
                    "wind": w_mult,
                    "hydro": h_mult
                }
    
    st.markdown("---")
    
    if st.button("🚀 Run Scenario Analysis"):
        scenario_data = latest.copy()
        
        # Apply national multipliers
        scenario_data["solar_capacity"] *= solar_mult
        scenario_data["wind_capacity"] *= wind_mult
        scenario_data["hydro_capacity"] *= hydro_mult
        
        # Apply state-specific multipliers
        for state, mults in state_multipliers.items():
            mask = scenario_data["state"] == state
            if "solar" in mults:
                scenario_data.loc[mask, "solar_capacity"] *= mults["solar"]
            if "wind" in mults:
                scenario_data.loc[mask, "wind_capacity"] *= mults["wind"]
            if "hydro" in mults:
                scenario_data.loc[mask, "hydro_capacity"] *= mults["hydro"]
        
        scenario_data["renewable_capacity"] = scenario_data["solar_capacity"] + scenario_data["wind_capacity"] + scenario_data["hydro_capacity"]
        scenario_data["supply_gap"] = scenario_data["peak_demand(mw)"] - scenario_data["renewable_capacity"]
        scenario_data["self_sufficiency_ratio"] = scenario_data["renewable_capacity"] / (scenario_data["peak_demand(mw)"] + 1e-6)
        
        st.markdown(f"## 📊 Results for {scenario_name}")
        st.markdown("---")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Supply Gap", f"{scenario_data['supply_gap'].mean():,.0f} MW", "After scenario")
        with col2:
            st.metric("Avg Self-Sufficiency", f"{scenario_data['self_sufficiency_ratio'].mean():.1%}", "After scenario")
        with col3:
            surplus = len(scenario_data[scenario_data["supply_gap"] < 0])
            st.metric("Surplus States", f"{surplus}/{len(scenario_data)}", "With positive surplus")
        with col4:
            critical = len(scenario_data[scenario_data["self_sufficiency_ratio"] < 0.2])
            st.metric("Critical States", f"{critical}/{len(scenario_data)}", "< 20% self-sufficiency")
        
        st.markdown("---")
        
        # State-by-state results
        st.markdown("### 📊 State-by-State Results")
        results_display = scenario_data[["state", "peak_demand(mw)", "renewable_capacity", "supply_gap", "self_sufficiency_ratio"]].copy()
        results_display.columns = ["State", "Peak Demand (MW)", "Renewable Supply (MW)", "Supply Gap (MW)", "Self-Sufficiency"]
        results_display["Self-Sufficiency"] = results_display["Self-Sufficiency"].apply(lambda x: f"{x:.1%}")
        results_display = results_display.sort_values("Supply Gap (MW)", ascending=False)
        
        st.dataframe(results_display, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Comparison chart
        st.markdown("### 📈 Self-Sufficiency Comparison")
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
            color_discrete_map={"Current": "#94A3B8", "After Scenario": "#10B981"}
        )
        fig_comparison.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_comparison, use_container_width=True)

# =====================================================
# PAGE: RANKINGS
# =====================================================
elif page == "📈 Rankings":
    st.markdown("# 📈 State Rankings & Status")
    
    latest = df.sort_values("year").groupby("state").last().reset_index()
    
    tab1, tab2, tab3, tab4 = st.tabs(["🌱 Self-Sufficiency", "⚡ Peak Demand", "☀️ Solar Efficiency", "⚠️ Supply Gap"])
    
    with tab1:
        st.markdown("## Renewable Energy Self-Sufficiency Ranking")
        ranking = latest[["state", "self_sufficiency_ratio"]].sort_values("self_sufficiency_ratio", ascending=False).reset_index(drop=True)
        ranking["Rank"] = range(1, len(ranking) + 1)
        st.dataframe(ranking[["Rank", "state", "self_sufficiency_ratio"]].rename(columns={"state": "State", "self_sufficiency_ratio": "Ratio"}), use_container_width=True, hide_index=True)
        
        fig = px.bar(ranking, x="state", y="self_sufficiency_ratio", color="self_sufficiency_ratio", color_continuous_scale="RdYlGn")
        fig.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("## Peak Demand Ranking")
        ranking = latest[["state", "peak_demand(mw)"]].sort_values("peak_demand(mw)", ascending=False).reset_index(drop=True)
        ranking["Rank"] = range(1, len(ranking) + 1)
        st.dataframe(ranking[["Rank", "state", "peak_demand(mw)"]].rename(columns={"state": "State", "peak_demand(mw)": "Demand (MW)"}), use_container_width=True, hide_index=True)
        
        fig = px.bar(ranking, x="state", y="peak_demand(mw)", color="peak_demand(mw)", color_continuous_scale="Reds")
        fig.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("## Solar Efficiency Ranking")
        ranking = latest[["state", "solar_efficiency"]].sort_values("solar_efficiency", ascending=False).reset_index(drop=True)
        ranking = ranking[ranking["solar_efficiency"].notna()]
        ranking["Rank"] = range(1, len(ranking) + 1)
        st.dataframe(ranking[["Rank", "state", "solar_efficiency"]].rename(columns={"state": "State", "solar_efficiency": "Efficiency"}), use_container_width=True, hide_index=True)
        
        fig = px.bar(ranking, x="state", y="solar_efficiency", color="solar_efficiency", color_continuous_scale="Oranges")
        fig.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("## Supply Gap Ranking")
        ranking = latest[["state", "supply_gap"]].sort_values("supply_gap", ascending=False).reset_index(drop=True)
        ranking["Rank"] = range(1, len(ranking) + 1)
        st.dataframe(ranking[["Rank", "state", "supply_gap"]].rename(columns={"state": "State", "supply_gap": "Gap (MW)"}), use_container_width=True, hide_index=True)
        
        colors = ["#EF4444" if x > 0 else "#10B981" for x in ranking["supply_gap"]]
        fig = go.Figure(data=[go.Bar(x=ranking["state"], y=ranking["supply_gap"], marker_color=colors)])
        fig.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; margin-top: 2rem;">
    <p>⚡ Energy Forecasting Dashboard v3.1 | ML-Powered Analytics</p>
    <p style="font-size: 0.85rem;">With Clustering, Forecasting & Scenario Analysis</p>
</div>
""", unsafe_allow_html=True)
