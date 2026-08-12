from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseDataInspector(ABC):
    """Abstract base class for data inspection."""

    @abstractmethod
    def summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """Generate a comprehensive summary of the DataFrame.

        Args:
            df: DataFrame to inspect.

        Returns:
            Dictionary containing summary statistics and metadata.
        """

    @abstractmethod
    def head(self, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """Return the first n rows.

        Args:
            df: DataFrame to inspect.
            n: Number of rows to return.

        Returns:
            DataFrame containing the first n rows.
        """

    @abstractmethod
    def tail(self, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """Return the last n rows.

        Args:
            df: DataFrame to inspect.
            n: Number of rows to return.

        Returns:
            DataFrame containing the last n rows.
        """

    @abstractmethod
    def shape(self, df: pd.DataFrame) -> tuple[int, int]:
        """Return the shape of the DataFrame.

        Args:
            df: DataFrame to inspect.

        Returns:
            Tuple of (rows, columns).
        """

    @abstractmethod
    def columns(self, df: pd.DataFrame) -> list[str]:
        """Return the column names.

        Args:
            df: DataFrame to inspect.

        Returns:
            List of column names.
        """

    @abstractmethod
    def dtypes(self, df: pd.DataFrame) -> dict[str, str]:
        """Return the data types of each column.

        Args:
            df: DataFrame to inspect.

        Returns:
            Dictionary mapping column names to data types.
        """

    @abstractmethod
    def memory_usage(self, df: pd.DataFrame) -> dict[str, int]:
        """Return memory usage per column.

        Args:
            df: DataFrame to inspect.

        Returns:
            Dictionary mapping column names to memory usage in bytes.
        """
