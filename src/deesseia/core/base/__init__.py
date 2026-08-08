"""Base abstract classes for core modules."""

from deesseia.core.base.cleaner import BaseCleaner
from deesseia.core.base.inspector import BaseDataInspector
from deesseia.core.base.loader import BaseDataLoader

__all__ = [
    "BaseCleaner",
    "BaseDataInspector",
    "BaseDataLoader",
]
