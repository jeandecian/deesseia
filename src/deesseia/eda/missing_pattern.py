from __future__ import annotations

from typing import Any, cast

import pandas as pd


class MissingPattern:
    """Missing value pattern analysis for data quality assessment."""

    EMPTY_DATAFRAME_ERROR: str = "DataFrame cannot be empty."

    def __init__(self) -> None:
        """Initialize the MissingPattern class."""

    def summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """Generate a summary of missing values in the DataFrame.

        Args:
            df: Input DataFrame.

        Returns:
            Dictionary containing missing value summary statistics.
        """

        if df.empty:
            raise ValueError(self.EMPTY_DATAFRAME_ERROR)

        total_cells: int = df.size
        total_missing: int = df.isna().sum().sum()
        total_complete: int = total_cells - total_missing

        missing_by_column: dict[str, int] = cast(
            dict[str, int], df.isna().sum().to_dict()
        )
        missing_percentage_by_column: dict[str, float] = cast(
            dict[str, float], (df.isna().sum() / len(df) * 100).to_dict()
        )

        columns_with_missing: list[str] = [
            col for col in df.columns if df[col].isna().any()
        ]

        columns_all_missing: list[str] = [
            col for col in df.columns if df[col].isna().all()
        ]

        complete_rows: int = df.dropna().shape[0]
        rows_with_missing: int = len(df) - complete_rows

        return {
            "total_cells": total_cells,
            "total_missing": total_missing,
            "total_complete": total_complete,
            "missing_percentage": (
                (total_missing / total_cells * 100) if total_cells > 0 else 0.0
            ),
            "complete_rows": complete_rows,
            "rows_with_missing": rows_with_missing,
            "missing_by_column": missing_by_column,
            "missing_percentage_by_column": missing_percentage_by_column,
            "columns_with_missing": columns_with_missing,
            "columns_all_missing": columns_all_missing,
            "total_columns_with_missing": len(columns_with_missing),
        }

    def pattern_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a missing pattern matrix showing which columns have missing values together.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame showing the missing pattern for each row.
        """

        if df.empty:
            raise ValueError(self.EMPTY_DATAFRAME_ERROR)

        missing_matrix: pd.DataFrame = df.isna().astype(int)

        missing_matrix["pattern"] = missing_matrix.astype(str).agg("".join, axis=1)

        pattern_counts: pd.DataFrame = (
            missing_matrix["pattern"]
            .value_counts()
            .reset_index(name="count")
            .rename(columns={"index": "pattern"})
        )

        cols: list[str] = df.columns.tolist()

        pattern_details: list[dict[str, Any]] = []
        for idx in range(len(pattern_counts)):
            pattern: str = str(pattern_counts.iloc[idx]["pattern"])
            count_value: int = int(pattern_counts.iloc[idx]["count"])

            pattern_dict: dict[str, Any] = {
                "pattern": pattern,
                "count": count_value,
            }

            for i, col in enumerate(cols):
                if i < len(pattern):
                    pattern_dict[col] = bool(int(pattern[i]))

            pattern_details.append(pattern_dict)

        result_df: pd.DataFrame = pd.DataFrame(pattern_details)

        cols_order: list[str] = ["pattern", "count"] + cols
        result_df = result_df[cols_order]

        return result_df

    def missing_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation between missing values in different columns.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame showing the correlation of missing values between columns.
        """

        if df.empty:
            raise ValueError(self.EMPTY_DATAFRAME_ERROR)

        missing_binary: pd.DataFrame = df.isna().astype(int)

        corr_matrix: pd.DataFrame = missing_binary.corr()

        return corr_matrix

    def missing_by_row(self, df: pd.DataFrame) -> pd.Series:
        """Count missing values per row.

        Args:
            df: Input DataFrame.

        Returns:
            Series with the number of missing values per row.
        """

        if df.empty:
            raise ValueError(self.EMPTY_DATAFRAME_ERROR)

        series: pd.Series = df.isna().sum(axis=1)

        return series

    def _check_correlation(
        self, col: str, other_col: str, missing_indicator: pd.Series, df: pd.DataFrame
    ) -> dict[str, Any] | None:
        """Check correlation between missingness in one column and values in another.

        Args:
            col: The column with missing values.
            other_col: The column to check correlation with.
            missing_indicator: Boolean indicator of missing values in col.
            df: Input DataFrame.

        Returns:
            Correlation dict if significant, None otherwise.
        """

        if not pd.api.types.is_numeric_dtype(df[other_col]):
            return None

        complete_cases: pd.Series = ~df[other_col].isna()

        if int(complete_cases.sum()) <= 1:
            return None

        corr: float = missing_indicator[complete_cases].corr(
            df[other_col][complete_cases]
        )

        if abs(corr) <= 0.3:
            return None

        return {
            "missing_col": col,
            "correlated_with": other_col,
            "correlation": float(corr),
        }

    def _find_correlations(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Find correlations between missing values and observed values.

        Args:
            df: Input DataFrame.

        Returns:
            List of correlations found.
        """

        cols_with_missing: list[str] = [
            col for col in df.columns if df[col].isna().any()
        ]

        correlations: list[dict[str, Any]] = []

        for col in cols_with_missing:
            missing_indicator: pd.Series = df[col].isna().astype(int)

            for other_col in df.columns:
                if other_col == col:
                    continue

                correlation: dict[str, Any] | None = self._check_correlation(
                    col, other_col, missing_indicator, df
                )

                if correlation is not None:
                    correlations.append(correlation)

        return correlations

    def _classify_missing_pattern(
        self, correlations: list[dict[str, Any]], df: pd.DataFrame
    ) -> tuple[str, str]:
        """Classify the missing data pattern based on correlations.

        Args:
            correlations: List of correlations found.
            df: Input DataFrame.

        Returns:
            Tuple of (pattern_type, evidence).
        """

        if len(correlations) == 0:
            row_missing: pd.Series = self.missing_by_row(df)

            variance: float = float(row_missing.var())
            mean: float = float(row_missing.mean())

            if variance < mean * 0.5:
                pattern_type: str = "MCAR (Missing Completely At Random)"
                evidence: str = (
                    "Missingness appears random with no strong correlations to observed values."
                )
            else:
                pattern_type = "MCAR (Missing Completely At Random) - weak evidence"
                evidence = "No strong correlations found, but missingness may not be completely random."

            return pattern_type, evidence

        pattern_type = "MAR (Missing At Random)"
        evidence = (
            f"Missingness is correlated with other observed variables: {correlations}"
        )

        return pattern_type, evidence

    def detect_pattern_type(self, df: pd.DataFrame) -> dict[str, Any]:
        """Detect the type of missing data pattern (MCAR, MAR, MNAR).

        This is a simplified detection based on correlations between missingness
        and observed values.

        Args:
            df: Input DataFrame.

        Returns:
            Dictionary with pattern type and supporting evidence.
        """

        if df.empty:
            raise ValueError(self.EMPTY_DATAFRAME_ERROR)

        if int(df.isna().sum().sum()) == 0:
            return {
                "pattern_type": "No missing values",
                "evidence": "No missing values found in the dataset.",
                "correlations_found": [],
            }

        correlations: list[dict[str, Any]] = self._find_correlations(df)
        missing_pattern: tuple[str, str] = self._classify_missing_pattern(
            correlations, df
        )
        pattern_type: str = missing_pattern[0]
        evidence: str = missing_pattern[1]

        return {
            "pattern_type": pattern_type,
            "evidence": evidence,
            "correlations_found": correlations,
        }

    def report(self, df: pd.DataFrame) -> dict[str, Any]:
        """Generate a comprehensive missing value report.

        Args:
            df: Input DataFrame.

        Returns:
            Dictionary containing complete missing value analysis.
        """

        if df.empty:
            raise ValueError(self.EMPTY_DATAFRAME_ERROR)

        summary: dict[str, Any] = self.summary(df)
        pattern_matrix: pd.DataFrame = self.pattern_matrix(df)
        missing_corr: pd.DataFrame = self.missing_correlation(df)
        pattern_type: dict[str, Any] = self.detect_pattern_type(df)

        row_missing: pd.Series = self.missing_by_row(df)
        top_missing_rows: dict[str, int] = cast(
            dict[str, int], row_missing.nlargest(5).to_dict()
        )

        missing_pct: pd.Series = df.isna().sum() / len(df) * 100
        top_missing_cols: dict[str, float] = cast(
            dict[str, float],
            missing_pct[missing_pct > 0].sort_values(ascending=False).head(5).to_dict(),
        )

        return {
            "summary": summary,
            "pattern_matrix": pattern_matrix,
            "missing_correlation": missing_corr,
            "pattern_type": pattern_type,
            "top_missing_rows": top_missing_rows,
            "top_missing_columns": top_missing_cols,
            "has_missing": bool(summary["total_missing"] > 0),
        }
