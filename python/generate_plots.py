"""
generate_plots.py
-----------------
Generates publication-quality static visualization figures for agricultural
yield forecasting using Random Forest regression.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def create_plots():
    # Setup directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "data", "sample_yield_weather.csv")
    plots_dir = os.path.join(base_dir, "..", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Load and prepare data
    df = pd.read_csv(data_path).sort_values("year").reset_index(drop=True)
    df["yield_lag1"] = df["yield_kg_ha"].shift(1)
    df["yield_lag2"] = df["yield_kg_ha"].shift(2)
    df["rainfall_ma2"] = df["rainfall_mm"].rolling(2).mean()
    df["temp_ma2"] = df["temp_mean_c"].rolling(2).mean()
    df = df.dropna().reset_index(drop=True)

    features = [
        "yield_lag1", "yield_lag2", "rainfall_mm", "temp_max_c",
        "temp_mean_c", "heat_stress_days", "solar_rad_mj",
        "ndvi_peak", "soil_moisture_pct", "rainfall_ma2", "temp_ma2"
    ]
    target = "yield_kg_ha"

    X = df[features].astype(float)
    y = df[target].astype(float)

    # Train Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    rf.fit(X, y)

    # 1. Ensemble Predictions and Standard Error from individual trees
    tree_preds = np.array([tree.predict(X.values) for tree in rf.estimators_])
    y_pred_mean = np.mean(tree_preds, axis=0)
    y_pred_std = np.std(tree_preds, axis=0)
    ci_lower = y_pred_mean - 1.96 * y_pred_std
    ci_upper = y_pred_mean + 1.96 * y_pred_std

    rmse = np.sqrt(mean_squared_error(y, y_pred_mean))
    r2 = r2_score(y, y_pred_mean)
    mape = np.mean(np.abs((y - y_pred_mean) / y)) * 100

    # Plot 1: Forecast Trajectory & Confidence Interval
    plt.figure(figsize=(10, 5), dpi=300)
    years = df["year"].values
    plt.plot(years, y.values, marker="o", color="#065f46", linewidth=2.2, label="Actual Observed Yield", zorder=3)
    plt.plot(years, y_pred_mean, marker="s", color="#0284c7", linewidth=2, linestyle="--", label="Random Forest Forecast", zorder=4)
    plt.fill_between(years, ci_lower, ci_upper, color="#38bdf8", alpha=0.25, label="95% Ensemble Confidence Interval")

    plt.title("Agricultural Crop Yield Forecasting: Actual vs. Random Forest Trajectory", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Cropping Season (Year)", fontsize=10, fontweight="bold")
    plt.ylabel("Crop Yield (kg/ha)", fontsize=10, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", loc="upper left")

    # Metrics annotation box
    metrics_text = f"Model Performance:\nRMSE: {rmse:.1f} kg/ha\nMAPE: {mape:.2f}%\nR²: {r2:.3f}"
    plt.annotate(metrics_text, xy=(0.80, 0.12), xycoords="axes fraction",
                 bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8fafc", edgecolor="#94a3b8"),
                 fontsize=9, family="monospace")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "forecast_trajectory.png"))
    plt.close()

    # Plot 2: Relative Feature Importance
    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)

    plt.figure(figsize=(9, 5), dpi=300)
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(importances)))
    bars = plt.barh(importances.index, importances.values, color=colors, edgecolor="#0f172a", height=0.65)
    plt.title("Relative Variable Importance in Agricultural Yield Prediction", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Mean Decrease in Impurity (Gini Importance)", fontsize=10, fontweight="bold")
    plt.grid(axis="x", linestyle=":", alpha=0.6)

    for bar in bars:
        val = bar.get_width()
        plt.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=8, color="#0f172a")

    plt.xlim(0, max(importances.values) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "feature_importance.png"))
    plt.close()

    # Plot 3: Partial Dependence Plots for Critical Weather Variables
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)
    PartialDependenceDisplay.from_estimator(
        rf, X, features=["rainfall_mm", "heat_stress_days"],
        ax=axes, line_kw={"color": "#047857", "linewidth": 2.5}
    )
    axes[0].set_title("Partial Dependence: Seasonal Rainfall", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("Seasonal Rainfall (mm)", fontsize=9)
    axes[0].set_ylabel("Partial Dependence (Yield kg/ha)", fontsize=9)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    axes[1].set_title("Partial Dependence: Heat Stress Days (>35°C)", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Heat Stress Days at Anthesis", fontsize=9)
    axes[1].set_ylabel("Partial Dependence (Yield kg/ha)", fontsize=9)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Non-linear Agricultural Weather Responses Captured by Random Forest", fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "partial_dependence.png"))
    plt.close()

    # Plot 4: Residuals Diagnostic
    residuals = y - y_pred_mean
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)

    ax1.scatter(y_pred_mean, residuals, color="#0284c7", edgecolor="#0f172a", s=40, alpha=0.85)
    ax1.axhline(0, color="#dc2626", linestyle="--", linewidth=1.5)
    ax1.set_title("Residuals vs. Fitted Yield", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Fitted Yield (kg/ha)", fontsize=9)
    ax1.set_ylabel("Residuals (kg/ha)", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.hist(residuals, bins=10, color="#10b981", edgecolor="#064e3b", alpha=0.85)
    ax2.axvline(0, color="#dc2626", linestyle="--", linewidth=1.5)
    ax2.set_title("Residuals Error Distribution", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Prediction Error (kg/ha)", fontsize=9)
    ax2.set_ylabel("Frequency", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "residuals_diagnostic.png"))
    plt.close()

    print(f"Successfully generated 4 static plots in: {plots_dir}")


if __name__ == "__main__":
    create_plots()
