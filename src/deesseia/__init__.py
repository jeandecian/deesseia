"""
Deesseia - Goddess of Data Science.

A comprehensive data science library that streamlines the entire data science workflow from data loading to model deployment.
"""

__version__ = "1.3.0"

from deesseia.core.base.cleaner import BaseCleaner
from deesseia.core.base.inspector import BaseDataInspector
from deesseia.core.base.loader import BaseDataLoader
from deesseia.core.base.validator import BaseValidator
from deesseia.core.cleaner import Cleaner
from deesseia.core.feature import FeatureCreator
from deesseia.core.inspector import DataInspector
from deesseia.core.loader import DataLoader
from deesseia.core.validator import Validator
from deesseia.eda.descriptive import DescriptiveStats
from deesseia.preprocess.encoder import Encoder
from deesseia.preprocess.imputer import Imputer
from deesseia.preprocess.scaler import Scaler
from deesseia.preprocess.split import Splitter
from deesseia.utils.fake_data import FakeDataGenerator

__all__ = [
    "BaseCleaner",
    "BaseDataInspector",
    "BaseDataLoader",
    "BaseValidator",
    "Cleaner",
    "DataInspector",
    "DataLoader",
    "DescriptiveStats",
    "Encoder",
    "FakeDataGenerator",
    "FeatureCreator",
    "Imputer",
    "Scaler",
    "Splitter",
    "Validator",
]
