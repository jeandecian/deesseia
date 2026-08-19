from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd


class Insights:
    """Generate automated insights from data."""

    def __init__(self) -> None:
        """Initialize the Insights."""

        self._params: dict[str, Any] = {}

    def generate(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate comprehensive automated insights from the DataFrame.

        Args:
            df: DataFrame to analyze.
            columns: List of columns to analyze. If None, analyzes all numeric columns.

        Returns:
            Dictionary containing automated insights.
        """

        if columns is None:
            cols: list[str] = df.select_dtypes(include=["number"]).columns.tolist()
        else:
            cols = [
                col
                for col in columns
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
            ]

        missing_analysis: dict[str, Any] = self._analyze_missing(df)

        if not cols:
            return {
                "skewed_columns": [],
                "outlier_columns": [],
                "high_correlation_pairs": [],
                "missing_analysis": missing_analysis,
                "recommendations": [],
            }

        insights: dict[str, Any] = {
            "skewed_columns": self._detect_skew(df, cols),
            "outlier_columns": self._detect_outliers(df, cols),
            "high_correlation_pairs": self._detect_high_correlation(df, cols),
            "missing_analysis": missing_analysis,
            "recommendations": self._generate_recommendations(df, cols),
        }

        return insights

    def _detect_skew(
        self,
        df: pd.DataFrame,
        cols: list[str],
        threshold: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Detect columns with significant skew.

        Args:
            df: DataFrame to analyze.
            cols: List of columns to analyze.
            threshold: Skewness threshold for detection.

        Returns:
            List of skewed columns with details.
        """

        skewed: list[dict[str, Any]] = []

        for col in cols:
            series: pd.Series = df[col].dropna()
            if series.empty:
                continue

            skew_value: float = cast(float, series.skew())

            if abs(skew_value) > threshold:
                skewed.append(
                    {
                        "column": col,
                        "skewness": skew_value,
                        "direction": "right" if skew_value > 0 else "left",
                        "severity": "high" if abs(skew_value) > 2 else "moderate",
                    }
                )

        return skewed

    def _detect_outliers(
        self,
        df: pd.DataFrame,
        cols: list[str],
    ) -> list[dict[str, Any]]:
        """Detect columns with outliers.

        Args:
            df: DataFrame to analyze.
            cols: List of columns to analyze.
            method: Method to detect outliers ('iqr' or 'zscore').

        Returns:
            List of columns with outlier information.
        """

        outliers: list[dict[str, Any]] = []

        for col in cols:
            series: pd.Series = df[col].dropna()
            if series.empty:
                continue

            q1: float = series.quantile(0.25)
            q3: float = series.quantile(0.75)
            iqr: float = q3 - q1
            lower_bound: float = q1 - 1.5 * iqr
            upper_bound: float = q3 + 1.5 * iqr
            outlier_count: int = int(
                ((series < lower_bound) | (series > upper_bound)).sum()
            )
            outlier_percentage: float = float(outlier_count / len(series) * 100)

            if outlier_count > 0:
                outliers.append(
                    {
                        "column": col,
                        "count": outlier_count,
                        "percentage": outlier_percentage,
                    }
                )

        return outliers

    def _detect_high_correlation(
        self,
        df: pd.DataFrame,
        cols: list[str],
        threshold: float = 0.8,
    ) -> list[dict[str, Any]]:
        """Detect highly correlated column pairs.

        Args:
            df: DataFrame to analyze.
            cols: List of columns to analyze.
            threshold: Correlation threshold for detection.

        Returns:
            List of high correlation pairs.
        """

        if len(cols) < 2:
            return []

        corr_matrix: pd.DataFrame = df[cols].corr().abs()
        high_corr: list[dict[str, Any]] = []

        upper: pd.DataFrame = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        for col1 in upper.columns:
            for col2 in upper.index:
                corr_value: Any = upper.loc[col1, col2]
                if pd.notna(corr_value) and corr_value > threshold:
                    high_corr.append(
                        {
                            "col1": col1,
                            "col2": col2,
                            "correlation": float(corr_value),
                        }
                    )

        high_corr.sort(key=lambda x: x["correlation"], reverse=True)

        return high_corr

    def _analyze_missing(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyze missing values in the DataFrame.

        Args:
            df: DataFrame to analyze.

        Returns:
            Dictionary containing missing value analysis.
        """

        missing_count: pd.Series = df.isnull().sum()
        missing_percentage: pd.Series = (missing_count / len(df)) * 100

        columns_with_missing: pd.Series = missing_count[missing_count > 0]

        if columns_with_missing.empty:
            return {
                "has_missing": False,
                "total_missing": 0,
                "columns": {},
            }

        return {
            "has_missing": True,
            "total_missing": int(missing_count.sum()),
            "columns": {
                col: {
                    "count": int(missing_count[col]),
                    "percentage": float(missing_percentage[col]),
                }
                for col in columns_with_missing.index
            },
        }

    def _generate_recommendations(
        self,
        df: pd.DataFrame,
        cols: list[str],
    ) -> list[str]:
        """Generate recommendations based on data analysis.

        Args:
            df: DataFrame to analyze.
            cols: List of numeric columns.

        Returns:
            List of recommendations.
        """

        recommendations: list[str] = []

        missing_analysis: dict[str, Any] = self._analyze_missing(df)
        if missing_analysis["has_missing"]:
            recommendations.append(
                f"Missing values detected in {len(missing_analysis['columns'])} column(s). "
                "Consider imputation or removal."
            )

        skewed: list[dict[str, Any]] = self._detect_skew(df, cols)
        if skewed:
            rec: str = "Skewed columns detected: "
            rec += ", ".join(
                [f"{s['column']} (skew: {s['skewness']:.2f})" for s in skewed]
            )
            rec += ". Consider log transformation."

            recommendations.append(rec)

        outliers: list[dict[str, Any]] = self._detect_outliers(df, cols)
        if outliers:
            rec = "Outliers detected in "
            rec += ", ".join(
                [
                    f"{o['column']} ({o['count']} outliers, {o['percentage']:.1f}%)"
                    for o in outliers
                ]
            )
            rec += ". Consider robust scaling or removal."

            recommendations.append(rec)

        high_corr: list[dict[str, Any]] = self._detect_high_correlation(df, cols)
        if high_corr:
            rec = "High correlation detected between: "
            rec += ", ".join(
                [
                    f"{h['col1']} & {h['col2']} (r={h['correlation']:.2f})"
                    for h in high_corr[:3]
                ]
            )

            if len(high_corr) > 3:
                rec += f" and {len(high_corr) - 3} more pair(s)."
            else:
                rec += "."

            rec += " Consider removing one feature from each pair."

            recommendations.append(rec)

        if not recommendations:
            recommendations.append("No significant issues detected. Data looks clean.")

        return recommendations
