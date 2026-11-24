"""
agrirf - Random Forest forecasting for agricultural yield time series.

Install (Python >= 3.8)::

    pip install "git+https://github.com/lunganlungpg/agri-rf-forecasting.git"

Quick use::

    import agrirf
    df = agrirf.load_sample_data()          # 35-season yield/weather dataset
    result = agrirf.run_pipeline()          # TimeSeriesSplit CV + feature importance
    agrirf.create_plots(output_dir="plots") # 300 DPI PNG figures
    agrirf.build_interactive_dashboard(output_dir="plots")  # Plotly HTML

The same functions are available from the command line as ``agri-rf``
(see ``agri-rf --help``).

Note: in the repository this package lives in the ``python/`` folder; the
``pyproject.toml`` maps that folder to the import name ``agrirf`` so the
existing scripts (``python python/agri_rf_forecast.py``) keep working.
"""

from .agri_rf_data import SAMPLE_URL, load_sample_data, resolve_data_path
from .agri_rf_forecast import (
    FEATURE_COLS,
    TARGET_COL,
    evaluate_forecast,
    load_and_preprocess,
    run_pipeline,
)
from .generate_interactive_plots import build_interactive_dashboard
from .generate_plots import create_plots

__version__ = "1.0.0"

__all__ = [
    "FEATURE_COLS",
    "TARGET_COL",
    "SAMPLE_URL",
    "load_sample_data",
    "resolve_data_path",
    "load_and_preprocess",
    "evaluate_forecast",
    "run_pipeline",
    "create_plots",
    "build_interactive_dashboard",
]
