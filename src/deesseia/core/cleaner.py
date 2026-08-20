from __future__ import annotations

import pandas as pd

from deesseia.core.base.cleaner import BaseCleaner


class Cleaner(BaseCleaner):
    """Clean and preprocess data."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize Cleaner with a DataFrame.

        Args:
            df: DataFrame to clean.
        """

        self._df: pd.DataFrame = df.copy()

    @property
    def df(self) -> pd.DataFrame:
        """Return the current DataFrame."""

        return self._df

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

        df: pd.DataFrame = self._df.copy()

        if columns is None:
            columns = df.select_dtypes(include=["number"]).columns.tolist()

        for col in columns:
            if col not in df.columns:
                continue

            if strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "mode":
                df[col] = df[col].fillna(
                    df[col].mode()[0] if not df[col].mode().empty else None
                )
            elif strategy == "drop":
                df = df.dropna(subset=[col])
            elif strategy == "ffill":
                df[col] = df[col].ffill()
            elif strategy == "bfill":
                df[col] = df[col].bfill()
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")

        self._df = df

        return df

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

        df: pd.DataFrame = self._df.copy()
        result: pd.DataFrame = df.drop_duplicates(subset=subset, keep=keep)  # type: ignore
        self._df = result

        return result

    def drop_columns(self, columns: list[str]) -> pd.DataFrame:
        """Drop specified columns from the DataFrame.

        Args:
            columns: List of column names to drop.

        Returns:
            DataFrame with specified columns removed.
        """

        df: pd.DataFrame = self._df.copy()
        result: pd.DataFrame = df.drop(columns=columns)
        self._df = result

        return result

    def drop_empty_columns(self) -> pd.DataFrame:
        """Drop columns that are completely empty (all nulls).

        Returns:
            DataFrame with empty columns removed.
        """

        df: pd.DataFrame = self._df.copy()
        result: pd.DataFrame = df.dropna(axis=1, how="all")
        self._df = result

        return result

    def drop_single_cardinality_columns(self) -> pd.DataFrame:
        """Drop columns with only one unique value.

        Returns:
            DataFrame with single-cardinality columns removed.
        """

        df: pd.DataFrame = self._df.copy()
        single_cardinality: list[str] = [
            col for col in df.columns if df[col].nunique() == 1
        ]
        result: pd.DataFrame = df.drop(columns=single_cardinality)
        self._df = result

        return result

    def standardize_column_names(self) -> pd.DataFrame:
        """Standardize column names to snake_case.

        Returns:
            DataFrame with standardized column names.
        """

        df: pd.DataFrame = self._df.copy()

        def clean_name(name: str) -> str:
            return name.lower().replace(" ", "_").replace("-", "_")

        df = df.rename(columns={col: clean_name(col) for col in df.columns})
        self._df = df

        return df

    def rename_columns(self, mapping: dict[str, str]) -> pd.DataFrame:
        """Rename columns.

        Args:
            mapping: Dictionary mapping old column names to new column names.

        Returns:
            DataFrame with renamed columns.
        """

        df: pd.DataFrame = self._df.copy()
        result: pd.DataFrame = df.rename(columns=mapping)
        self._df = result

        return result

    def select_columns(self, columns: list[str]) -> pd.DataFrame:
        """Select specified columns.

        Args:
            columns: List of column names to select.

        Returns:
            DataFrame with only the selected columns.
        """

        df: pd.DataFrame = self._df.copy()
        result: pd.DataFrame = df.loc[:, columns]
        self._df = result

        return result
