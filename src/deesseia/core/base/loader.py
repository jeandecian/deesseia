from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseDataLoader(ABC):
    """Abstract base class for data loading."""

    @abstractmethod
    def load(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Load data from a source into a DataFrame.

        Args:
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.

        Returns:
            DataFrame containing the loaded data.
        """
