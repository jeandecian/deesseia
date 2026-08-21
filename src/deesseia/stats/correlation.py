from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd


class Correlation:
    """Correlation analysis for data exploration."""

    def __init__(self) -> None:
        """Initialize the Correlation."""

        self._params: dict[str, Any] = {}

    def correlation_matrix(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        method: Literal["pearson", "spearman", "kendall"] = "pearson",
    ) -> pd.DataFrame:
        """Generate correlation matrix.

        Args:
            df: DataFrame to analyze.
            columns: List of columns to include. If None, uses all numeric columns.
            method: Correlation method ('pearson', 'spearman', 'kendall').

        Returns:
            Correlation matrix as DataFrame.
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
            empty_df: pd.DataFrame = pd.DataFrame()

            return empty_df

        corr_matrix: pd.DataFrame = df[cols].corr(method=method)

        return corr_matrix

    def high_correlations(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        method: Literal["pearson", "spearman", "kendall"] = "pearson",
        threshold: float = 0.8,
    ) -> list[dict[str, Any]]:
        """Find highly correlated column pairs.

        Args:
            df: DataFrame to analyze.
            columns: List of columns to include. If None, uses all numeric columns.
            method: Correlation method ('pearson', 'spearman', 'kendall').
            threshold: Correlation threshold for detection.

        Returns:
            List of high correlation pairs sorted by correlation descending.
        """

        if columns is None:
            cols: list[str] = df.select_dtypes(include=["number"]).columns.tolist()
        else:
            cols = [
                col
                for col in columns
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
            ]

        if len(cols) < 2:
            return []

        corr_matrix: pd.DataFrame = df[cols].corr(method=method).abs()
        high_corr: list[dict[str, Any]] = []

        upper: pd.DataFrame = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        for col1 in upper.columns:
            for col2 in upper.index:
                corr_value: Any = upper.loc[col1, col2]
                if pd.notna(corr_value):
                    corr_val_float: float = float(corr_value)
                    if corr_val_float > threshold:
                        high_corr.append(
                            {
                                "col1": col1,
                                "col2": col2,
                                "correlation": corr_val_float,
                            }
                        )

        high_corr.sort(key=lambda x: x["correlation"], reverse=True)

        return high_corr

    def plot_heatmap(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        method: Literal["pearson", "spearman", "kendall"] = "pearson",
        figsize: tuple[int, int] = (10, 8),
        annot: bool = True,
        cmap: str = "RdBu_r",
        **kwargs: Any,
    ) -> None:
        """Plot correlation matrix as a heatmap.

        Args:
            df: DataFrame to analyze.
            columns: List of columns to include. If None, uses all numeric columns.
            method: Correlation method ('pearson', 'spearman', 'kendall').
            figsize: Figure size (width, height).
            annot: Whether to annotate correlation values.
            cmap: Colormap for the heatmap.
            **kwargs: Additional arguments passed to seaborn.heatmap.
        """

        import matplotlib.pyplot as plt
        import seaborn as sns

        corr_matrix: pd.DataFrame = self.correlation_matrix(
            df, columns=columns, method=method
        )

        if corr_matrix.empty:
            print("No numeric columns found for correlation heatmap.")
            return

        plt.figure(figsize=figsize)  # type: ignore
        sns.heatmap(  # type: ignore
            corr_matrix,
            annot=annot,
            cmap=cmap,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            **kwargs,
        )
        plt.title(f"Correlation Matrix ({method.capitalize()})", fontsize=14)  # type: ignore
        plt.tight_layout()
        plt.show()  # type: ignore
