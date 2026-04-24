# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_excel("enriched_data.xlsx")
df.columns = df.columns.str.strip()
df = df.sort_values(["state", "year"]).reset_index(drop=True)

# =====================================================
# FEATURE ENGINEERING
# =====================================================
df["renewable_capacity"] = (
    df["solar_capacity"].fillna(0) +
    df["wind_capacity"].fillna(0) +
    df["hydro_capacity"].fillna(0)
)

df["supply_gap"] = df["peak_demand(mw)"] - df["renewable_capacity"]

df["self_sufficiency_ratio"] = df["renewable_capacity"] / (df["peak_demand(mw)"] + 1e-6)

df["solar_efficiency"] = df["solar_capacity"] / (df["solar_irradiance"] + 1e-6)

# =====================================================
# LAGS (TIME SERIES SAFE - DO NOT TOUCH)
# =====================================================
df["lag_1"] = df.groupby("state")["peak_demand(mw)"].shift(1).bfill()
df["lag_2"] = df.groupby("state")["peak_demand(mw)"].shift(2).bfill()

# =====================================================
# CORRELATION HEATMAP
# =====================================================
num_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(12,8))
sns.heatmap(num_df.corr(), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()

print("\n🔹 Correlation with Peak Demand:\n")
print(num_df.corr()["peak_demand(mw)"].sort_values(ascending=False))

# =====================================================
# FEATURE SELECTION
# =====================================================
demand_features = [
    "energy_consumption",
    "population",
    "solar_capacity",
    "wind_capacity",
    "solar_utilisation_rate",
    "wind_utilisation_rate",
    "temp_x_consumption",
    "prev_year_capacity",
    "td_loss_pct",
    "lag_1",
    "lag_2"
]

supply_features = [
    "solar_capacity",
    "wind_capacity",
    "hydro_capacity",
    "solar_utilisation_rate",
    "wind_utilisation_rate"
]

demand_features = [f for f in demand_features if f in df.columns]
supply_features = [f for f in supply_features if f in df.columns]

# =====================================================
# TRAIN / TEST SPLIT
# =====================================================
train = df[(df["year"] >= 2020) & (df["year"] <= 2023)]
test  = df[df["year"] == 2024]

# =====================================================
# ================= DEMAND MODEL ======================
# =====================================================
X_train_d = train[demand_features]
y_train_d = train["peak_demand(mw)"]

X_test_d = test[demand_features]
y_test_d = test["peak_demand(mw)"]

rf_d = RandomForestRegressor(n_estimators=300, random_state=42)
lr_d = LinearRegression()

rf_d.fit(X_train_d, y_train_d)
lr_d.fit(X_train_d, y_train_d)

rf_pred_d = rf_d.predict(X_test_d)
lr_pred_d = lr_d.predict(X_test_d)

print("\n================ DEMAND MODEL =================\n")

print("Random Forest")
print("R2  :", r2_score(y_test_d, rf_pred_d))
print("MAE :", mean_absolute_error(y_test_d, rf_pred_d))
print("RMSE:", np.sqrt(mean_squared_error(y_test_d, rf_pred_d)))

print("\nLinear Regression (baseline)")
print("R2  :", r2_score(y_test_d, lr_pred_d))
print("MAE :", mean_absolute_error(y_test_d, lr_pred_d))
print("RMSE:", np.sqrt(mean_squared_error(y_test_d, lr_pred_d)))

# =====================================================
# FEATURE IMPORTANCE (RF)
# =====================================================
imp_d = pd.Series(rf_d.feature_importances_, index=demand_features).sort_values(ascending=False)

print("\n🔥 Demand Feature Importance:\n")
print(imp_d)

plt.figure()
imp_d.plot(kind="bar")
plt.title("Demand Feature Importance")
plt.show()

# =====================================================
# ================= SUPPLY MODEL ======================
# =====================================================
X_train_s = train[supply_features]
y_train_s = train["renewable_capacity"]

X_test_s = test[supply_features]
y_test_s = test["renewable_capacity"]

rf_s = RandomForestRegressor(n_estimators=300, random_state=42)
lr_s = LinearRegression()

rf_s.fit(X_train_s, y_train_s)
lr_s.fit(X_train_s, y_train_s)

rf_pred_s = rf_s.predict(X_test_s)
lr_pred_s = lr_s.predict(X_test_s)

print("\n================ SUPPLY MODEL =================\n")

print("Random Forest")
print("R2  :", r2_score(y_test_s, rf_pred_s))
print("MAE :", mean_absolute_error(y_test_s, rf_pred_s))
print("RMSE:", np.sqrt(mean_squared_error(y_test_s, rf_pred_s)))

print("\nLinear Regression (baseline)")
print("R2  :", r2_score(y_test_s, lr_pred_s))
print("RMSE:", np.sqrt(mean_squared_error(y_test_s, lr_pred_s)))

# =====================================================
# FEATURE IMPORTANCE (SUPPLY)
# =====================================================
imp_s = pd.Series(rf_s.feature_importances_, index=supply_features).sort_values(ascending=False)

print("\n🔥 Supply Feature Importance:\n")
print(imp_s)

plt.figure()
imp_s.plot(kind="bar")
plt.title("Supply Feature Importance")
plt.show()

# =====================================================
# SUPPLY GAP
# =====================================================
gap = rf_pred_d - rf_pred_s

print("\n================ SUPPLY GAP =================\n")
print(gap[:10])

# =====================================================
# CLUSTERING
# =====================================================
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
df["cluster"] = kmeans.fit_predict(scaled)

cluster_rank = df.groupby("cluster")["energy_consumption"].mean().sort_values().index.tolist()

label_map = {
    cluster_rank[0]: "Low Infrastructure",
    cluster_rank[1]: "Moderate Growth",
    cluster_rank[2]: "High Demand Emerging",
    cluster_rank[3]: "Energy Intensive Leaders"
}

df["cluster_label"] = df["cluster"].map(label_map)

print("\n================ CLUSTERS =================\n")
print(df.groupby("state")["cluster_label"].first().reset_index())

# =====================================================
# RANKINGS
# =====================================================
latest = df.sort_values("year").groupby("state").last().reset_index()

print("\n🏆 SELF SUFFICIENT STATES:\n")
print(latest.sort_values("self_sufficiency_ratio", ascending=False)[
    ["state", "self_sufficiency_ratio"]
].head(10))

print("\n⚠️ SUPPLY GAP STATES:\n")
print(latest.sort_values("supply_gap", ascending=False)[
    ["state", "supply_gap"]
].head(10))

print("\n☀️ SOLAR EFFICIENCY STATES:\n")
print(latest.sort_values("solar_efficiency", ascending=False)[
    ["state", "solar_efficiency"]
].head(10))

# =====================================================
# CLASSIFICATION
# =====================================================
def classify(x):
    if x > 0.6:
        return "Surplus"
    elif x > 0.4:
        return "On Track"
    elif x > 0.2:
        return "At Risk"
    return "Critical"

latest["status"] = latest["self_sufficiency_ratio"].apply(classify)

print("\n================ STATE STATUS =================\n")
print(latest[["state", "status"]])

# =====================================================
# SCENARIO FUNCTION (FOR FRONTEND SIMULATION)
# =====================================================
def run_scenario(df,
                 solar_mult=1.0,
                 wind_mult=1.0,
                 hydro_mult=1.0,
                 state_overrides=None):

    sc = df.copy()

    sc["solar_capacity"] *= solar_mult
    sc["wind_capacity"] *= wind_mult
    sc["hydro_capacity"] *= hydro_mult

    if state_overrides:
        for state, ch in state_overrides.items():
            m = sc["state"] == state

            if "solar" in ch:
                sc.loc[m, "solar_capacity"] *= ch["solar"]
            if "wind" in ch:
                sc.loc[m, "wind_capacity"] *= ch["wind"]
            if "hydro" in ch:
                sc.loc[m, "hydro_capacity"] *= ch["hydro"]

    sc["renewable_capacity"] = (
        sc["solar_capacity"] +
        sc["wind_capacity"] +
        sc["hydro_capacity"]
    )

    sc["new_gap"] = sc["peak_demand(mw)"] - sc["renewable_capacity"]

    return sc

# =====================================================
# FUTURE FORECAST (2025–2030)
# =====================================================
future_years = range(2025, 2031)
future_rows = []

for state in df["state"].unique():

    last = df[df["state"] == state].iloc[-1].copy()

    for y in future_years:
        row = last.copy()
        row["year"] = y

        # growth assumptions
        row["energy_consumption"] *= 1.03
        row["population"] *= 1.01

        row["solar_capacity"] *= 1.04
        row["wind_capacity"] *= 1.035
        row["hydro_capacity"] *= 1.01

        # keep realism: NO lag recalculation here (intentional)

        future_rows.append(row)

future_df = pd.DataFrame(future_rows)

# =====================================================
# BASE PREDICTIONS
# =====================================================
future_df["pred_demand"] = rf_d.predict(future_df[demand_features])
future_df["pred_supply"] = rf_s.predict(future_df[supply_features])

# =====================================================
# DEFAULT GAP
# =====================================================
future_df["pred_gap"] = future_df["pred_demand"] - future_df["pred_supply"]

# =====================================================
# SCENARIO APPLICATION (IMPORTANT ADDITION)
# =====================================================
def apply_future_scenario(df_future, solar=1.0, wind=1.0, hydro=1.0):

    sc = df_future.copy()

    sc["solar_capacity"] *= solar
    sc["wind_capacity"] *= wind
    sc["hydro_capacity"] *= hydro

    sc["pred_supply"] = rf_s.predict(sc[supply_features])
    sc["pred_gap"] = sc["pred_demand"] - sc["pred_supply"]

    return sc

# example scenario (can be replaced by frontend inputs)
scenario_df = apply_future_scenario(future_df, solar=1.2, wind=1.1, hydro=1.05)

print("\n================ FUTURE FORECAST =================\n")
print(scenario_df[["state", "year", "pred_demand", "pred_supply", "pred_gap"]].head(20))

# =====================================================
# PLOT
# =====================================================
plt.figure()
plt.plot(scenario_df.groupby("year")["pred_demand"].mean(), label="Demand")
plt.plot(scenario_df.groupby("year")["pred_supply"].mean(), label="Supply")
plt.plot(scenario_df.groupby("year")["pred_gap"].mean(), label="Gap")

plt.legend()
plt.title("India Energy Forecast (Scenario Applied)")
plt.show()