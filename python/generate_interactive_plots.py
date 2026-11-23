"""
generate_interactive_plots.py
------------------------------
Generates a standalone, self-contained interactive Plotly HTML dashboard
for agricultural crop yield forecasting with Random Forest.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor


def build_interactive_dashboard():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "data", "sample_yield_weather.csv")
    plots_dir = os.path.join(base_dir, "..", "plots")
    os.makedirs(plots_dir, exist_ok=True)

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

    rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    rf.fit(X, y)

    # Predictions & tree bounds
    tree_preds = np.array([tree.predict(X.values) for tree in rf.estimators_])
    y_pred = np.mean(tree_preds, axis=0)
    y_std = np.std(tree_preds, axis=0)
    upper_bound = y_pred + 1.96 * y_std
    lower_bound = y_pred - 1.96 * y_std
    years = df["year"].tolist()

    # 1. Main Dashboard Figure (2x2 Grid)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "<b>Actual vs. Forecasted Crop Yield (Interactive Trajectory)</b>",
            "<b>Variable Importance Ranking (MDI)</b>",
            "<b>Residuals vs. Fitted Yield (Error Diagnostics)</b>",
            "<b>Agro-Climatic Correlation Matrix</b>"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.10
    )

    # Subplot (1,1): Trajectory & Uncertainty Band
    fig.add_trace(
        go.Scatter(
            x=years + years[::-1],
            y=upper_bound.tolist() + lower_bound.tolist()[::-1],
            fill="toself",
            fillcolor="rgba(16, 185, 129, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="95% Confidence Band"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=years, y=y,
            mode="lines+markers",
            name="Actual Yield",
            line=dict(color="#065f46", width=2.5),
            marker=dict(size=7, color="#047857"),
            hovertemplate="<b>Year</b>: %{x}<br><b>Actual Yield</b>: %{y:.1f} kg/ha<extra></extra>"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=years, y=y_pred,
            mode="lines+markers",
            name="RF Forecast",
            line=dict(color="#0284c7", width=2, dash="dash"),
            marker=dict(size=6, color="#0284c7", symbol="square"),
            hovertemplate="<b>Year</b>: %{x}<br><b>Predicted Yield</b>: %{y:.1f} kg/ha<extra></extra>"
        ),
        row=1, col=1
    )

    # Subplot (1,2): Variable Importance Horizontal Bar Chart
    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
    fig.add_trace(
        go.Bar(
            x=importances.values,
            y=importances.index,
            orientation="h",
            name="Importance",
            marker=dict(
                color=importances.values,
                colorscale="Viridis",
                line=dict(color="#0f172a", width=1)
            ),
            hovertemplate="<b>Feature</b>: %{y}<br><b>Importance Score</b>: %{x:.4f}<extra></extra>"
        ),
        row=1, col=2
    )

    # Subplot (2,1): Residuals vs Fitted
    residuals = y - y_pred
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode="markers",
            name="Residuals",
            marker=dict(
                size=8,
                color="#0284c7",
                line=dict(color="#0f172a", width=1),
                opacity=0.85
            ),
            hovertemplate="<b>Fitted</b>: %{x:.1f} kg/ha<br><b>Residual</b>: %{y:.1f} kg/ha<extra></extra>"
        ),
        row=2, col=1
    )
    # Zero error line
    fig.add_shape(
        type="line",
        x0=min(y_pred), x1=max(y_pred), y0=0, y1=0,
        line=dict(color="#dc2626", dash="dot", width=1.5),
        row=2, col=1
    )

    # Subplot (2,2): Correlation Matrix Heatmap
    corr_vars = ["yield_kg_ha", "rainfall_mm", "temp_max_c", "heat_stress_days", "solar_rad_mj", "ndvi_peak", "soil_moisture_pct"]
    corr_matrix = df[corr_vars].corr().round(2)
    fig.add_trace(
        go.Heatmap(
            z=corr_matrix.values,
            x=corr_vars,
            y=corr_vars,
            colorscale="Tealgrn",
            text=corr_matrix.values,
            texttemplate="%{text}",
            name="Correlation",
            hoverongaps=False
        ),
        row=2, col=2
    )

    # Global Layout Configuration
    fig.update_layout(
        title=dict(
            text="🌾 <b>Interactive Agricultural Forecasting Dashboard | Random Forest Analytics</b>",
            font=dict(size=18, color="#0f172a"),
            x=0.05
        ),
        template="plotly_white",
        height=850,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_xaxes(title_text="Cropping Year", row=1, col=1)
    fig.update_yaxes(title_text="Yield (kg/ha)", row=1, col=1)

    fig.update_xaxes(title_text="Mean Decrease in Impurity", row=1, col=2)

    fig.update_xaxes(title_text="Fitted Yield (kg/ha)", row=2, col=1)
    fig.update_yaxes(title_text="Residual Error (kg/ha)", row=2, col=1)

    # Save multi-panel dashboard HTML
    dashboard_path = os.path.join(plots_dir, "interactive_dashboard.html")
    fig.write_html(dashboard_path, include_plotlyjs="cdn")

    # 2. Interactive 3D Crop-Climate Response Surface
    # Grid for Rainfall and Heat Stress Days
    rain_grid = np.linspace(df["rainfall_mm"].min(), df["rainfall_mm"].max(), 25)
    heat_grid = np.linspace(df["heat_stress_days"].min(), df["heat_stress_days"].max(), 25)
    R_mesh, H_mesh = np.meshgrid(rain_grid, heat_grid)

    # Average baseline for other features
    baseline = X.mean().to_dict()
    surface_rows = []
    for r_val, h_val in zip(R_mesh.ravel(), H_mesh.ravel()):
        row = baseline.copy()
        row["rainfall_mm"] = r_val
        row["heat_stress_days"] = h_val
        surface_rows.append(row)

    Z_pred = rf.predict(pd.DataFrame(surface_rows)[features]).reshape(R_mesh.shape)

    fig_3d = go.Figure(data=[
        go.Surface(
            z=Z_pred,
            x=rain_grid,
            y=heat_grid,
            colorscale="Viridis",
            colorbar=dict(title="Predicted Yield (kg/ha)")
        )
    ])
    fig_3d.update_layout(
        title="<b>Interactive 3D Response Surface: Crop Yield vs. Rainfall and Heat Stress</b>",
        scene=dict(
            xaxis_title="Seasonal Rainfall (mm)",
            yaxis_title="Heat Stress Days (>35°C)",
            zaxis_title="Yield (kg/ha)"
        ),
        height=700,
        template="plotly_white"
    )

    surface_path = os.path.join(plots_dir, "interactive_3d_surface.html")
    fig_3d.write_html(surface_path, include_plotlyjs="cdn")

    print(f"Interactive plots successfully generated:\n  - {dashboard_path}\n  - {surface_path}")


if __name__ == "__main__":
    build_interactive_dashboard()
