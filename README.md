# Agricultural Forecasting with Random Forest

Hands-on code examples in Python and R demonstrating how to apply Random Forest regression to agricultural yield forecasting using time-lagged features and weather observations.

## Project Structure

- `data/sample_yield_weather.csv`: Historical crop yield and weather dataset (1995-2024).
- `python/agri_rf_forecast.py`: Python workflow with lag feature creation and time-series cross-validation.
- `python/requirements.txt`: Python package requirements.
- `R/agri_rf_forecast.R`: R script using the `randomForest` package.

## Getting Started

### Python
Install dependencies and run:
```bash
pip install -r python/requirements.txt
python python/agri_rf_forecast.py
```

### R
Run the R script:
```bash
Rscript R/agri_rf_forecast.R
```

## Dataset Fields
- `year`: Cropping season year
- `yield_kg_ha`: Final harvest yield
- `rainfall_mm`: Total seasonal rainfall
- `temp_mean_c`: Mean growing temperature
- `solar_rad_mj`: Average solar radiation
- `ndvi_peak`: Peak vegetation index
- `soil_moisture_pct`: Average soil moisture percentage
