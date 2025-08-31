"""FastDatasets public API.

This package re-exports the core classes from the internal `app` package
to provide a stable import path for users when installed via PyPI.
"""

from app.core.config import Config, config  # noqa: F401
from app.core.document import DocumentProcessor  # noqa: F401
from app.core.dataset import DatasetBuilder  # noqa: F401
from .api import generate_dataset, generate_dataset_to_dir  # noqa: F401

__all__ = [
    "Config",
    "config",
    "DocumentProcessor",
    "DatasetBuilder",
    "generate_dataset",
    "generate_dataset_to_dir",
]




