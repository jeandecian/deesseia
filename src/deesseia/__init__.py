"""
Deesseia - Goddess of Data Science.

A comprehensive data science library that streamlines the entire data science workflow from data loading to model deployment.
"""

__version__ = "0.0.0"

from deesseia.core.base.cleaner import BaseCleaner
from deesseia.core.base.inspector import BaseDataInspector
from deesseia.core.base.loader import BaseDataLoader
from deesseia.core.base.validator import BaseValidator
from deesseia.core.inspector import DataInspector
from deesseia.core.loader import DataLoader
from deesseia.utils.fake_data import FakeDataGenerator

__all__ = [
    "BaseCleaner",
    "BaseDataInspector",
    "BaseDataLoader",
    "BaseValidator",
    "DataInspector",
    "DataLoader",
    "FakeDataGenerator",
]
