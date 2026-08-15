from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class Scaler:
    """Scale and normalize features for machine learning."""

    def __init__(self) -> None:
        """Initialize the Scaler."""

        self._fitted: bool = False
        self._params: dict[str, Any] = {}

    def minmax_scale(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        feature_range: tuple[float, float] = (0, 1),
    ) -> pd.DataFrame:
        """Scale features to a given range, default [0, 1].

        Args:
            df: DataFrame to scale.
            columns: List of columns to scale. If None, scales all numeric columns.
            feature_range: Desired range of transformed data (min, max).

        Returns:
            Scaled DataFrame.
        """

        result: pd.DataFrame = df.copy()
        cols: list[str] = (
            columns or df.select_dtypes(include=["number"]).columns.tolist()
        )

        for col in cols:
            if col not in df.columns:
                continue

            min_val: float = df[col].min()
            max_val: float = df[col].max()

            if max_val == min_val:
                result[col] = 0.0
            else:
                result[col] = (df[col] - min_val) / (max_val - min_val) * (
                    feature_range[1] - feature_range[0]
                ) + feature_range[0]

        return result

    def standard_scale(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Standardize features by removing the mean and scaling to unit variance.

        Args:
            df: DataFrame to scale.
            columns: List of columns to scale. If None, scales all numeric columns.

        Returns:
            Standardized DataFrame.
        """

        result: pd.DataFrame = df.copy()
        cols: list[str] = (
            columns or df.select_dtypes(include=["number"]).columns.tolist()
        )

        for col in cols:
            if col not in df.columns:
                continue

            mean: float = df[col].mean()
            std: float = df[col].std()

            if std == 0:
                result[col] = 0.0
            else:
                result[col] = (df[col] - mean) / std

        return result

    def robust_scale(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Scale features using statistics that are robust to outliers (median and IQR).

        Args:
            df: DataFrame to scale.
            columns: List of columns to scale. If None, scales all numeric columns.

        Returns:
            Robustly scaled DataFrame.
        """

        result: pd.DataFrame = df.copy()
        cols: list[str] = (
            columns or df.select_dtypes(include=["number"]).columns.tolist()
        )

        for col in cols:
            if col not in df.columns:
                continue

            median: float = df[col].median()
            q1: float = df[col].quantile(0.25)
            q3: float = df[col].quantile(0.75)
            iqr: float = q3 - q1

            if iqr == 0:
                result[col] = 0.0
            else:
                result[col] = (df[col] - median) / iqr

        return result

    def maxabs_scale(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Scale features by their maximum absolute value (range [-1, 1]).

        Args:
            df: DataFrame to scale.
            columns: List of columns to scale. If None, scales all numeric columns.

        Returns:
            MaxAbs scaled DataFrame.
        """

        result: pd.DataFrame = df.copy()
        cols: list[str] = (
            columns or df.select_dtypes(include=["number"]).columns.tolist()
        )

        for col in cols:
            if col not in df.columns:
                continue

            max_abs: float = df[col].abs().max()

            if max_abs == 0:
                result[col] = 0.0
            else:
                result[col] = df[col] / max_abs

        return result

    def log_transform(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        base: float = np.e,
    ) -> pd.DataFrame:
        """Apply logarithmic transformation to features.

        Args:
            df: DataFrame to transform.
            columns: List of columns to transform. If None, transforms all numeric columns.
            base: Logarithm base. Default is e (natural log).

        Returns:
            Log-transformed DataFrame.
        """
        result: pd.DataFrame = df.copy()
        cols: list[str] = (
            columns or df.select_dtypes(include=["number"]).columns.tolist()
        )

        for col in cols:
            if col not in df.columns:
                continue

            # Add small epsilon to avoid log(0)
            min_val: float = df[col].min()

            if min_val <= 0:
                epsilon: float = abs(min_val) + 1e-10
                result[col] = np.log(df[col] + epsilon) / np.log(base)
            else:
                result[col] = np.log(df[col]) / np.log(base)

        return result
