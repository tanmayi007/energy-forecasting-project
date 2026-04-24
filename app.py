# -*- coding: utf-8 -*-
"""
Energy Forecasting Dashboard
Interactive Streamlit frontend for ML energy prediction model
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="⚡ Energy Forecast Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for refined aesthetics
st.markdown("""
    <style>
    /* Typography & Theme */
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Space Mono', monospace;
        letter-spacing: -0.5px;
    }
    
    /* Color Scheme */
    :root {
        --primary: #00D084;
        --secondary: #FF6B6B;
        --dark: #0F1419;
        --light: #F7FAFC;
        --accent: #4F46E5;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #0F1419 0%, #1A202C 100%);
    }
    
    .main {
        background: linear-gradient(135deg, #FAFBFC 0%, #F0F4F8 100%);
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #00D084;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
    }
    
    /* Button styling */
    button {
        background: linear-gradient(135deg, #00D084 0%, #00B86D 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 12px rgba(0, 208, 132, 0.3) !important;
    }
    
    /* Card containers */
    .forecast-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
        border-top: 3px solid #00D084;
    }
    
    .status-surplus {
        color: #10B981;
        font-weight: 700;
    }
    
    .status-ontrack {
        color: #F59E0B;
        font-weight: 700;
    }
    
    .status-atrisk {
        color: #EF4444;
        font-weight: 700;
    }
    
    .status-critical {
        color: #991B1B;
        font-weight: 700;
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
        # Import the model script
        import sys
        sys.path.append('.')
        
        # Load data
        df = pd.read_excel("enriched_data.xlsx")
        df.columns = df.columns.str.strip()
        df = df.sort_values(["state", "year"]).reset_index(drop=True)
        
        # Feature engineering (same as model)
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

# Load data
df = load_model_and_data()

if df is None:
    st.error("Unable to load data. Please ensure 'enriched_data.xlsx' is in the working directory.")
    st.stop()

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
st.sidebar.markdown("## ⚡ Navigation")
page = st.sidebar.radio(
    "Select View",
    ["🏠 Dashboard", "📊 Analytics", "🔮 Forecast", "⚙️ Scenarios", "📈 Rankings"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
Energy Forecasting dashboard using Random Forest models to predict:
- Peak electricity demand
- Renewable energy supply
- Supply-demand gaps

**Last Updated:** 2024
""")

# =====================================================
# PAGE: DASHBOARD
# =====================================================
if page == "🏠 Dashboard":
    st.markdown("# ⚡ Energy Forecast Hub")
    st.markdown("Real-time energy demand and supply analytics for India")
    
    # Get latest data
    latest = df.sort_values("year").groupby("state").last().reset_index()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_demand = latest["peak_demand(mw)"].mean()
        st.metric(
            "Avg Peak Demand",
            f"{avg_demand:,.0f} MW",
            f"across {len(latest)} states"
        )
    
    with col2:
        avg_supply = latest["renewable_capacity"].mean()
        st.metric(
            "Avg Renewable Supply",
            f"{avg_supply:,.0f} MW",
            "current capacity"
        )
    
    with col3:
        avg_gap = latest["supply_gap"].mean()
        gap_color = "🟢" if avg_gap < 0 else "🔴"
        st.metric(
            "Avg Supply Gap",
            f"{avg_gap:,.0f} MW",
            gap_color
        )
    
    with col4:
        avg_sufficiency = latest["self_sufficiency_ratio"].mean()
        st.metric(
            "Avg Self-Sufficiency",
            f"{avg_sufficiency:.1%}",
            "renewable capacity ratio"
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
            marker_color="#FF6B6B"
        ))
        fig_dist.add_trace(go.Box(
            y=latest["renewable_capacity"],
            name="Renewable Supply",
            marker_color="#00D084"
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
        
        fig_gap = go.Figure(data=[
            go.Bar(
                x=gap_data["state"],
                y=gap_data["supply_gap"],
                marker_color=["#FF6B6B" if x > 0 else "#00D084" for x in gap_data["supply_gap"]],
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
        st.markdown(f"### {selected_state}")
        st.metric("Peak Demand", f"{state_data['peak_demand(mw)']:,.0f} MW")
        st.metric("Solar Capacity", f"{state_data['solar_capacity']:,.0f} MW")
    
    with col2:
        st.metric("Wind Capacity", f"{state_data['wind_capacity']:,.0f} MW")
        st.metric("Hydro Capacity", f"{state_data['hydro_capacity']:,.0f} MW")
    
    with col3:
        status_class = state_data["status"] if hasattr(state_data, "status") else "On Track"
        st.metric("Self-Sufficiency", f"{state_data['self_sufficiency_ratio']:.1%}")
        st.markdown(f"**Status:** {status_class}")

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
                marker_color="#4F46E5"
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
                marker_color="#FF6B6B"
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
        line=dict(color="#FF6B6B", width=3)
    ))
    
    fig_trend.add_trace(go.Scatter(
        x=time_series["year"],
        y=time_series["renewable_capacity"],
        mode='lines+markers',
        name='Avg Renewable Capacity',
        line=dict(color="#00D084", width=3)
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
        line=dict(color="#FF6B6B", width=2, dash="dash")
    ))
    
    fig_forecast.add_trace(go.Scatter(
        x=national_forecast["year"],
        y=national_forecast["peak_demand(mw)"],
        mode='lines+markers',
        name='Forecast Demand',
        line=dict(color="#FF6B6B", width=3)
    ))
    
    fig_forecast.add_trace(go.Scatter(
        x=historical["year"],
        y=historical["renewable_capacity"],
        mode='lines+markers',
        name='Historical Supply',
        line=dict(color="#00D084", width=2, dash="dash")
    ))
    
    fig_forecast.add_trace(go.Scatter(
        x=national_forecast["year"],
        y=national_forecast["renewable_capacity"],
        mode='lines+markers',
        name='Forecast Supply',
        line=dict(color="#00D084", width=3)
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
        
        colors = ["#FF6B6B" if x > 0 else "#00D084" for x in ranking["supply_gap"]]
        
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
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>⚡ Energy Forecasting Dashboard | Built with Streamlit & ML</p>
    <p style="font-size: 0.85rem;">Data-driven insights for sustainable energy planning</p>
</div>
""", unsafe_allow_html=True)