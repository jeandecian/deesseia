from __future__ import annotations

from typing import Any, cast

import pandas as pd


class DescriptiveStats:
    """Generate descriptive statistics for data exploration."""

    def __init__(self) -> None:
        """Initialize the DescriptiveStats."""

        self._params: dict[str, Any] = {}

    def summary(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate comprehensive summary statistics.

        Args:
            df: DataFrame to analyze.
            columns: List of columns to analyze. If None, analyzes all numeric columns.

        Returns:
            Dictionary containing summary statistics.
        """

        if columns is None:
            cols: list[str] = df.select_dtypes(include=["number"]).columns.tolist()
        else:
            cols = [
                col
                for col in columns
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
            ]

        if not cols:
            return {}

        stats: dict[str, Any] = {}

        for col in cols:
            series: pd.Series = df[col].dropna()
            if series.empty:
                continue

            stats[col] = {
                "count": int(series.count()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "mode": list(series.mode()),
                "std": float(series.std()),
                "var": float(series.var()),
                "min": float(series.min()),
                "max": float(series.max()),
                "q1": float(series.quantile(0.25)),
                "q2": float(series.quantile(0.50)),
                "q3": float(series.quantile(0.75)),
                "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
                "skew": cast(float, series.skew()),
                "kurtosis": cast(float, series.kurtosis()),
                "missing_count": int(df[col].isna().sum()),
                "missing_percentage": float(df[col].isna().sum() / len(df) * 100),
            }

        return stats

    def five_number_summary(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate five-number summary (min, Q1, median, Q3, max).

        Args:
            df: DataFrame to analyze.
            columns: List of columns to analyze. If None, analyzes all numeric columns.

        Returns:
            Dictionary containing five-number summaries.
        """

        if columns is None:
            cols: list[str] = df.select_dtypes(include=["number"]).columns.tolist()
        else:
            cols = [
                col
                for col in columns
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
            ]

        if not cols:
            return {}

        summary: dict[str, Any] = {}

        for col in cols:
            series: pd.Series = df[col].dropna()
            if series.empty:
                continue

            summary[col] = {
                "min": float(series.min()),
                "q1": float(series.quantile(0.25)),
                "median": float(series.median()),
                "q3": float(series.quantile(0.75)),
                "max": float(series.max()),
            }

        return summary
