from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseCleaner(ABC):
    """Abstract base class for data cleaning operations."""

    @abstractmethod
    def handle_missing(
        self,
        strategy: str,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Handle missing values in the DataFrame.

        Args:
            strategy: Strategy to use ('mean', 'median', 'mode', 'drop', 'ffill', 'bfill').
            columns: Optional list of columns to apply the strategy to.
                     If None, applies to all numeric columns.

        Returns:
            DataFrame with missing values handled.
        """

    @abstractmethod
    def drop_duplicates(
        self,
        subset: list[str] | None = None,
        keep: str = "first",
    ) -> pd.DataFrame:
        """Drop duplicate rows from the DataFrame.

        Args:
            subset: Optional list of columns to consider for duplicates.
            keep: Which duplicates to keep ('first', 'last', False).

        Returns:
            DataFrame with duplicates removed.
        """

    @abstractmethod
    def drop_columns(self, columns: list[str]) -> pd.DataFrame:
        """Drop specified columns from the DataFrame.

        Args:
            columns: List of column names to drop.

        Returns:
            DataFrame with specified columns removed.
        """

    @abstractmethod
    def drop_empty_columns(self) -> pd.DataFrame:
        """Drop columns that are completely empty (all nulls).

        Returns:
            DataFrame with empty columns removed.
        """

    @abstractmethod
    def drop_single_cardinality_columns(self) -> pd.DataFrame:
        """Drop columns with only one unique value.

        Returns:
            DataFrame with single-cardinality columns removed.
        """
