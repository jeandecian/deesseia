from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseValidator(ABC):
    """Abstract base class for data validation."""

    @abstractmethod
    def validate_schema(
        self,
        df: pd.DataFrame,
        schema: dict[str, str],
    ) -> dict[str, list[str]]:
        """Validate DataFrame against a schema.

        Args:
            df: DataFrame to validate.
            schema: Dictionary mapping column names to expected data types.

        Returns:
            Dictionary with 'missing_columns' and 'wrong_dtypes' lists.
        """

    @abstractmethod
    def check_missing_values(self, df: pd.DataFrame) -> dict[str, float]:
        """Check for missing values in each column.

        Args:
            df: DataFrame to check.

        Returns:
            Dictionary mapping column names to missing percentage.
        """

    @abstractmethod
    def check_duplicates(self, df: pd.DataFrame) -> dict[str, int]:
        """Check for duplicate rows.

        Args:
            df: DataFrame to check.

        Returns:
            Dictionary with 'total_duplicates' and count per column if subset specified.
        """
