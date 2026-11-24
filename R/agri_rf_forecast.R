# agri_rf_forecast.R
# ------------------
# R implementation of Random Forest for agricultural yield forecasting
# using lagged features and environmental predictors.
#
# This file is part of the `agrirf` R package:
#   remotes::install_github("lunganlungpg/agri-rf-forecasting")
#   agrirf::rf_run_pipeline()
#
# It can also still be run as a plain script from the repository root:
#   Rscript R/agri_rf_forecast.R

#' Public URL of the sample yield/weather dataset
#'
#' Used when the package is installed without a local copy of the repository.
#' @keywords internal
rf_sample_url <- function() {
  "https://raw.githubusercontent.com/lunganlungpg/agri-rf-forecasting/main/data/sample_yield_weather.csv"
}

#' Locate the sample dataset
#'
#' Looks for `data/sample_yield_weather.csv` in (or one level above) the
#' working directory, i.e. inside a cloned copy of the repository. If it is not
#' found, the public GitHub copy is used instead.
#'
#' @return A file path or URL that [utils::read.csv()] can read.
#' @export
#' @examples
#' rf_sample_data_path()
rf_sample_data_path <- function() {
  candidates <- c(file.path("data", "sample_yield_weather.csv"),
                  file.path("..", "data", "sample_yield_weather.csv"))
  found <- candidates[file.exists(candidates)]
  if (length(found)) found[[1]] else rf_sample_url()
}

#' Load the sample 35-season yield/weather dataset
#'
#' @param path Optional CSV path or URL. Defaults to [rf_sample_data_path()].
#' @return A data frame sorted by `year`.
#' @export
#' @examples
#' \donttest{
#' df <- rf_load_sample_data()
#' head(df)
#' }
rf_load_sample_data <- function(path = rf_sample_data_path()) {
  df <- utils::read.csv(path)
  df[order(df$year), ]
}

#' Add autoregressive yield lags
#'
#' Creates `yield_lag1` and `yield_lag2` (yield in the previous one and two
#' seasons) and drops the first rows that have no lag available.
#'
#' @param df Data frame with `year` and `yield_kg_ha` columns.
#' @return The data frame with lag columns and complete cases only.
#' @export
rf_add_lag_features <- function(df) {
  df <- df[order(df$year), ]
  n <- nrow(df)
  df$yield_lag1 <- c(NA, df$yield_kg_ha[seq_len(n - 1)])
  df$yield_lag2 <- c(NA, NA, df$yield_kg_ha[seq_len(n - 2)])
  stats::na.omit(df)
}

#' Fit the Random Forest yield forecaster
#'
#' @param df Data frame as returned by [rf_load_sample_data()] (lags are
#'   added automatically).
#' @param ntree Number of trees.
#' @param mtry Number of predictors tried at each split.
#' @param seed Random seed for reproducibility.
#' @return A `randomForest` model fitted with `importance = TRUE`.
#' @export
#' @examples
#' \donttest{
#' model <- rf_fit_forecast(rf_load_sample_data())
#' randomForest::importance(model)
#' }
rf_fit_forecast <- function(df = rf_load_sample_data(), ntree = 200, mtry = 3, seed = 42) {
  model_data <- rf_add_lag_features(df)
  set.seed(seed)
  formula_rf <- yield_kg_ha ~ yield_lag1 + yield_lag2 + rainfall_mm +
    temp_mean_c + solar_rad_mj + ndvi_peak + soil_moisture_pct
  randomForest::randomForest(
    formula = formula_rf,
    data = model_data,
    ntree = ntree,
    mtry = mtry,
    importance = TRUE
  )
}

#' Run the full forecasting demo and print the results
#'
#' Fits the model, prints the summary and the variable importance ranked by
#' `%IncMSE` (same output as the original script).
#'
#' @param path CSV path or URL. Defaults to [rf_sample_data_path()].
#' @return Invisibly, a list with `model` and `importance`.
#' @export
#' @examples
#' \donttest{
#' rf_run_pipeline()
#' }
rf_run_pipeline <- function(path = rf_sample_data_path()) {
  rf_model <- rf_fit_forecast(rf_load_sample_data(path))
  print(rf_model)

  cat("\nVariable Importance (%IncMSE):\n")
  importance_df <- randomForest::importance(rf_model)
  ranked <- importance_df[order(-importance_df[, "%IncMSE"]), , drop = FALSE]
  print(ranked)
  invisible(list(model = rf_model, importance = ranked))
}

# Script mode: `Rscript R/agri_rf_forecast.R` runs the demo. When the file is
# loaded as part of the installed package, sys.nframe() is > 0, so nothing runs.
if (sys.nframe() == 0L) {
  if (!requireNamespace("randomForest", quietly = TRUE)) {
    install.packages("randomForest", repos = "https://cloud.r-project.org")
  }
  rf_run_pipeline()
}
