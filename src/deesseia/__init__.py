"""
Deesseia - Goddess of Data Science.

A comprehensive data science library that streamlines the entire data science workflow from data loading to model deployment.
"""

__version__ = "0.0.0"

from deesseia.core.base.inspector import BaseDataInspector
from deesseia.core.base.loader import BaseDataLoader

__all__ = [
    "BaseDataInspector",
    "BaseDataLoader",
]
