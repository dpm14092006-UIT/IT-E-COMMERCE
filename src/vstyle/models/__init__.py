
"""Active forecasting model registry."""

TABULAR_MODELS = ("random_forest", "lightgbm", "catboost")
SEQUENCE_MODELS = ("tcn",)
FORECAST_MODELS = ("tcn", "catboost", "lightgbm", "random_forest")
