from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import stats


class Distributions:
    """Generate distribution analysis for data exploration."""

    def __init__(self) -> None:
        """Initialize the Distributions."""

        self._params: dict[str, Any] = {}

    def histogram(
        self,
        df: pd.DataFrame,
        column: str,
        bins: int = 10,
    ) -> dict[str, Any]:
        """Generate histogram data for a column.

        Args:
            df: DataFrame to analyze.
            column: Column name to analyze.
            bins: Number of bins for the histogram.

        Returns:
            Dictionary containing histogram data.
        """

        if column not in df.columns:
            return {}

        series: pd.Series = df[column].dropna()
        if series.empty:
            return {}

        histogram: tuple[np.ndarray, np.ndarray] = np.histogram(series, bins=bins)
        hist: np.ndarray = histogram[0]
        bin_edges: np.ndarray = histogram[1]

        return {
            "column": column,
            "count": int(series.count()),
            "hist": hist.tolist(),
            "bin_edges": bin_edges.tolist(),
            "bins": bins,
        }

    def kde(
        self,
        df: pd.DataFrame,
        column: str,
        bandwidth: float | None = None,
        n_points: int = 100,
    ) -> dict[str, Any]:
        """Generate kernel density estimation data for a column.

        Args:
            df: DataFrame to analyze.
            column: Column name to analyze.
            bandwidth: Bandwidth for KDE. If None, uses scott's rule.
            n_points: Number of points for the KDE curve.

        Returns:
            Dictionary containing KDE data.
        """

        if column not in df.columns:
            return {}

        series: pd.Series = df[column].dropna()
        if series.empty:
            return {}

        if bandwidth is None:
            bandwidth = stats.gaussian_kde(series).scotts_factor() * series.std(ddof=1)

        kde: stats.gaussian_kde = stats.gaussian_kde(series, bw_method=bandwidth)
        x_min: float = series.min()
        x_max: float = series.max()
        x: np.ndarray = np.linspace(x_min, x_max, n_points)
        y: np.ndarray = kde(x)

        return {
            "column": column,
            "x": x.tolist(),
            "y": y.tolist(),
            "bandwidth": float(bandwidth),
            "n_points": n_points,
        }

    def boxplot(
        self,
        df: pd.DataFrame,
        column: str,
    ) -> dict[str, Any]:
        """Generate boxplot statistics for a column.

        Args:
            df: DataFrame to analyze.
            column: Column name to analyze.

        Returns:
            Dictionary containing boxplot statistics.
        """

        if column not in df.columns:
            return {}

        series: pd.Series = df[column].dropna()
        if series.empty:
            return {}

        q1: float = series.quantile(0.25)
        q3: float = series.quantile(0.75)
        iqr: float = q3 - q1
        lower_whisker: float = max(series.min(), q1 - 1.5 * iqr)
        upper_whisker: float = min(series.max(), q3 + 1.5 * iqr)

        return {
            "column": column,
            "min": float(series.min()),
            "q1": float(q1),
            "median": float(series.median()),
            "q3": float(q3),
            "max": float(series.max()),
            "iqr": float(iqr),
            "lower_whisker": float(lower_whisker),
            "upper_whisker": float(upper_whisker),
            "outliers": series[
                (series < lower_whisker) | (series > upper_whisker)
            ].tolist(),
            "count": int(series.count()),
        }

    def violinplot(
        self,
        df: pd.DataFrame,
        column: str,
        n_points: int = 100,
    ) -> dict[str, Any]:
        """Generate violin plot data for a column.

        Args:
            df: DataFrame to analyze.
            column: Column name to analyze.
            n_points: Number of points for the violin curve.

        Returns:
            Dictionary containing violin plot data.
        """

        if column not in df.columns:
            return {}

        series: pd.Series = df[column].dropna()
        if series.empty:
            return {}

        # Use gaussian KDE for violin
        kde: stats.gaussian_kde = stats.gaussian_kde(series)
        x_min: float = series.min()
        x_max: float = series.max()
        x: np.ndarray = np.linspace(x_min, x_max, n_points)
        y: np.ndarray = kde(x)

        # Scale to fit violin width
        max_y: float = y.max()
        if max_y > 0:
            y = y / max_y

        return {
            "column": column,
            "x": x.tolist(),
            "y": y.tolist(),
            "min": float(series.min()),
            "median": float(series.median()),
            "max": float(series.max()),
            "count": int(series.count()),
            "n_points": n_points,
        }

    def summary(
        self,
        df: pd.DataFrame,
        column: str,
        bins: int = 10,
    ) -> dict[str, Any]:
        """Generate a complete distribution summary for a column.

        Args:
            df: DataFrame to analyze.
            column: Column name to analyze.
            bins: Number of bins for the histogram.

        Returns:
            Dictionary containing all distribution data.
        """

        if column not in df.columns:
            return {}

        series: pd.Series = df[column].dropna()
        if series.empty:
            return {}

        # Histogram
        histogram: tuple[np.ndarray, np.ndarray] = np.histogram(series, bins=bins)
        hist: np.ndarray = histogram[0]
        bin_edges: np.ndarray = histogram[1]

        # KDE
        kde: stats.gaussian_kde = stats.gaussian_kde(series)
        x_min: float = series.min()
        x_max: float = series.max()
        x: np.ndarray = np.linspace(x_min, x_max, 100)
        y: np.ndarray = kde(x)

        # Boxplot
        q1: float = series.quantile(0.25)
        q3: float = series.quantile(0.75)
        iqr: float = q3 - q1
        lower_whisker: float = max(series.min(), q1 - 1.5 * iqr)
        upper_whisker: float = min(series.max(), q3 + 1.5 * iqr)

        return {
            "column": column,
            "count": int(series.count()),
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "skew": cast(float, series.skew()),
            "kurtosis": cast(float, series.kurtosis()),
            "histogram": {
                "hist": hist.tolist(),
                "bin_edges": bin_edges.tolist(),
                "bins": bins,
            },
            "kde": {
                "x": x.tolist(),
                "y": y.tolist(),
            },
            "boxplot": {
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "lower_whisker": float(lower_whisker),
                "upper_whisker": float(upper_whisker),
                "outliers": series[
                    (series < lower_whisker) | (series > upper_whisker)
                ].tolist(),
            },
        }
