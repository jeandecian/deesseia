from __future__ import annotations

from typing import Any, cast

import pandas as pd

from deesseia.core.base.inspector import BaseDataInspector


class DataInspector(BaseDataInspector):
    """Inspect and profile dataframes."""

    def summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """Generate a comprehensive summary of the DataFrame.

        Args:
            df: DataFrame to inspect.

        Returns:
            Dictionary containing summary statistics and metadata.
        """

        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "memory_usage": df.memory_usage(deep=True).to_dict(),
            "total_memory": df.memory_usage(deep=True).sum(),
            "describe": df.describe(include="all").to_dict(),
        }

    def head(self, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """Return the first n rows.

        Args:
            df: DataFrame to inspect.
            n: Number of rows to return.

        Returns:
            DataFrame containing the first n rows.
        """

        first_n_rows: pd.DataFrame = df.head(n)

        return first_n_rows

    def tail(self, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """Return the last n rows.

        Args:
            df: DataFrame to inspect.
            n: Number of rows to return.

        Returns:
            DataFrame containing the last n rows.
        """

        last_n_rows: pd.DataFrame = df.tail(n)

        return last_n_rows

    def shape(self, df: pd.DataFrame) -> tuple[int, int]:
        """Return the shape of the DataFrame.

        Args:
            df: DataFrame to inspect.

        Returns:
            Tuple of (rows, columns).
        """

        return df.shape

    def columns(self, df: pd.DataFrame) -> list[str]:
        """Return the column names.

        Args:
            df: DataFrame to inspect.

        Returns:
            List of column names.
        """

        return df.columns.tolist()

    def dtypes(self, df: pd.DataFrame) -> dict[str, str]:
        """Return the data types of each column.

        Args:
            df: DataFrame to inspect.

        Returns:
            Dictionary mapping column names to data types.
        """

        return cast(dict[str, str], df.dtypes.to_dict())

    def memory_usage(self, df: pd.DataFrame) -> dict[str, int]:
        """Return memory usage per column.

        Args:
            df: DataFrame to inspect.

        Returns:
            Dictionary mapping column names to memory usage in bytes.
        """

        return cast(dict[str, int], df.memory_usage(deep=True).to_dict())
