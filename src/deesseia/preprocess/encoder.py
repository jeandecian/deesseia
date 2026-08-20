from __future__ import annotations

from typing import Any

import pandas as pd


class Encoder:
    """Encode categorical features for machine learning."""

    def __init__(self) -> None:
        """Initialize the Encoder."""
        self._fitted: bool = False
        self._mappings: dict[str, dict[Any, int]] = {}
        self._categories: dict[str, list[Any]] = {}

    def _get_categorical_columns(self, df: pd.DataFrame) -> list[str]:
        """Get categorical columns (object, category, or string dtype)."""

        return list(df.select_dtypes(include=["object", "category", "string"]).columns)

    def label_encode(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Encode categorical labels with values between 0 and n_classes-1.

        Args:
            df: DataFrame to encode.
            columns: List of columns to encode. If None, encodes all object/categorical columns.

        Returns:
            DataFrame with encoded columns.
        """

        result: pd.DataFrame = df.copy()
        cols: list[str] = columns or self._get_categorical_columns(df)

        for col in cols:
            if col not in df.columns:
                continue

            # Get unique values
            unique_vals: list[Any] = df[col].dropna().unique().tolist()
            mapping: dict[Any, int] = {
                val: idx for idx, val in enumerate(sorted(unique_vals))
            }

            # Apply mapping, handling None/NaN
            result[col] = df[col].map(mapping)

            # Store mapping for potential inverse transform
            self._mappings[col] = mapping

        return result

    def one_hot_encode(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        drop_first: bool = False,
    ) -> pd.DataFrame:
        """Encode categorical features as a one-hot numeric array.

        Args:
            df: DataFrame to encode.
            columns: List of columns to encode. If None, encodes all object/categorical columns.
            drop_first: Whether to drop the first category to avoid multicollinearity.

        Returns:
            DataFrame with one-hot encoded columns.
        """

        result: pd.DataFrame = df.copy()
        cols: list[str] = columns or self._get_categorical_columns(df)

        # Convert to one-hot
        for col in cols:
            if col not in df.columns:
                continue

            # Get dummies
            dummies: pd.DataFrame = pd.get_dummies(
                df[col], prefix=col, drop_first=drop_first, dtype=int
            )

            # Drop the original column and join the dummies
            result = result.drop(columns=[col])
            result = pd.concat([result, dummies], axis=1)

        return result

    def target_encode(
        self,
        df: pd.DataFrame,
        column: str,
        target: str,
        min_samples_leaf: int = 1,
        smoothing: float = 1.0,
    ) -> pd.DataFrame:
        """Encode a categorical column using the target mean (with optional smoothing).

        Args:
            df: DataFrame to encode.
            column: Categorical column to encode.
            target: Target column to compute means from.
            min_samples_leaf: Minimum samples per category for smoothing.
            smoothing: Smoothing parameter for target encoding.

        Returns:
            DataFrame with target-encoded column.
        """

        result: pd.DataFrame = df.copy()

        if column not in df.columns:
            return result

        if target not in df.columns:
            return result

        # Calculate global target mean
        global_mean: float = df[target].mean()

        # Calculate per-category target mean and count
        category_stats: pd.DataFrame = df.groupby(column)[target].agg(["mean", "count"])
        category_stats = category_stats.rename(
            columns={"mean": "mean", "count": "count"}
        )

        # Apply smoothing
        def smooth_mean(count: float, mean: float) -> float:
            if count >= min_samples_leaf:
                return mean
            else:
                # Shrink towards global mean
                weight: float = count / (count + smoothing)
                return weight * mean + (1 - weight) * global_mean

        def apply_smoothing(row: pd.Series) -> float:
            return smooth_mean(float(row["count"]), float(row["mean"]))

        category_stats["encoded"] = category_stats.apply(apply_smoothing, axis=1)

        # Map encoded values back to the DataFrame
        result[column] = result[column].map(category_stats["encoded"])

        return result

    def frequency_encode(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """Encode categorical features by frequency count.

        Args:
            df: DataFrame to encode.
            columns: List of columns to encode. If None, encodes all object/categorical columns.
            normalize: Whether to normalize frequencies to [0, 1].

        Returns:
            DataFrame with frequency-encoded columns.
        """

        result: pd.DataFrame = df.copy()
        cols: list[str] = columns or self._get_categorical_columns(df)

        for col in cols:
            if col not in df.columns:
                continue

            # Calculate frequencies
            if normalize:
                freq: pd.Series = df[col].value_counts(normalize=True)
            else:
                freq = df[col].value_counts(normalize=False)

            result[col] = df[col].map(freq)

        return result

    def ordinal_encode(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        ordering: dict[str, list[Any]] | None = None,
    ) -> pd.DataFrame:
        """Encode ordinal categorical features with custom ordering.

        Args:
            df: DataFrame to encode.
            columns: List of columns to encode. If None, encodes all object/categorical columns.
            ordering: Dictionary mapping column names to ordered lists of categories.

        Returns:
            DataFrame with ordinal-encoded columns.
        """

        result: pd.DataFrame = df.copy()
        cols: list[str] = columns or self._get_categorical_columns(df)

        for col in cols:
            if col not in df.columns:
                continue

            if ordering and col in ordering:
                # Use custom ordering
                ord_mapping: dict[Any, int] = {
                    val: idx for idx, val in enumerate(ordering[col])
                }
                result[col] = df[col].map(ord_mapping)
            else:
                # Fall back to label encoding
                unique_vals: list[Any] = df[col].dropna().unique().tolist()
                label_mapping: dict[Any, int] = {
                    val: idx for idx, val in enumerate(sorted(unique_vals))
                }
                result[col] = df[col].map(label_mapping)

        return result
