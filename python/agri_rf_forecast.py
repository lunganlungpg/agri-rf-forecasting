"""
agri_rf_forecast.py
-------------------
Practical demonstration of Random Forest Regressor for agricultural yield forecasting
using lagged time-series features and climatic variables.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """Load agricultural time series and engineer lagged/rolling features."""
    df = pd.read_csv(filepath)
    df = df.sort_values("year").reset_index(drop=True)

    # 1. Autoregressive lags (Yield from previous years)
    df["yield_lag1"] = df["yield_kg_ha"].shift(1)
    df["yield_lag2"] = df["yield_kg_ha"].shift(2)

    # 2. Moving average of climatic variables (2-year moving window)
    df["rainfall_ma2"] = df["rainfall_mm"].rolling(window=2).mean()
    df["temp_ma2"] = df["temp_mean_c"].rolling(window=2).mean()

    # Drop initial rows with NaN values created by lagging
    df = df.dropna().reset_index(drop=True)
    return df


def evaluate_forecast(y_true, y_pred) -> dict:
    """Calculate standard statistical metrics used in crop forecasting."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "MAPE (%)": mape, "R2": r2}


def run_pipeline():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "..", "data", "sample_yield_weather.csv")

    df = load_and_preprocess(data_path)

    # Define predictor features and target variable
    feature_cols = [
        "yield_lag1",
        "yield_lag2",
        "rainfall_mm",
        "temp_mean_c",
        "solar_rad_mj",
        "ndvi_peak",
        "soil_moisture_pct",
        "rainfall_ma2",
        "temp_ma2",
    ]
    target_col = "yield_kg_ha"

    X = df[feature_cols]
    y = df[target_col]

    # TimeSeriesSplit cross-validation to prevent future-data leakage
    tscv = TimeSeriesSplit(n_splits=5)
    fold = 1
    fold_metrics = []

    print("=" * 65)
    print("Random Forest Agricultural Forecasting - Time Series Evaluation")
    print("=" * 65)

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_split=2,
            random_state=42,
        )
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)

        m = evaluate_forecast(y_test, preds)
        fold_metrics.append(m)
        print(f"Fold {fold} | Test Years: {df.loc[test_idx, 'year'].values} | RMSE: {m['RMSE']:.2f} kg/ha | MAPE: {m['MAPE (%)']:.2f}%")
        fold += 1

    # Fit final model on all data for feature importance
    final_rf = RandomForestRegressor(n_estimators=150, max_depth=5, random_state=42)
    final_rf.fit(X, y)

    importances = pd.Series(final_rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    print("\n" + "=" * 65)
    print("Relative Feature Importance (Mean Decrease in Impurity):")
    print("=" * 65)
    for feat, imp in importances.items():
        print(f"  - {feat:<20}: {imp:.4f}")


if __name__ == "__main__":
    run_pipeline()
