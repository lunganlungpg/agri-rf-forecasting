# agri_rf_forecast.R
# ------------------
# R implementation of Random Forest for agricultural yield forecasting
# using lagged features and environmental predictors.

suppressPackageStartupMessages({
  if (!requireNamespace("randomForest", quietly = TRUE)) {
    install.packages("randomForest", repos = "https://cloud.r-project.org")
  }
  library(randomForest)
})

# 1. Load dataset
data_path <- file.path("..", "data", "sample_yield_weather.csv")
if (!file.exists(data_path)) {
  data_path <- file.path("data", "sample_yield_weather.csv")
}

df <- read.csv(data_path)
df <- df[order(df$year), ]

# 2. Engineer Lagged Features for Time Series
n <- nrow(df)
df$yield_lag1 <- c(NA, df$yield_kg_ha[1:(n - 1)])
df$yield_lag2 <- c(NA, NA, df$yield_kg_ha[1:(n - 2)])

# Complete cases after lagging
model_data <- na.omit(df)

# 3. Fit Random Forest Model
set.seed(42)
formula_rf <- yield_kg_ha ~ yield_lag1 + yield_lag2 + rainfall_mm + 
              temp_mean_c + solar_rad_mj + ndvi_peak + soil_moisture_pct

rf_model <- randomForest(
  formula = formula_rf,
  data = model_data,
  ntree = 200,
  mtry = 3,
  importance = TRUE
)

# 4. Model Summary & Evaluation
print(rf_model)

# 5. Extract Variable Importance
cat("\nVariable Importance (%IncMSE):\n")
importance_df <- importance(rf_model)
print(importance_df[order(-importance_df[, "%IncMSE"]), , drop = FALSE])
