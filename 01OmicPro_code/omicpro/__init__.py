"""OmicPro: multi-omics prediction based on prompt learning with incomplete data."""

from .model import MultiOmicsMultiTaskRegressor
from .model_registry import available_models, build_model

__all__ = ["MultiOmicsMultiTaskRegressor", "available_models", "build_model"]
__version__ = "0.2.0"
