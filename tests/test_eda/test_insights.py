from typing import Any

import pandas as pd
import pytest  # noqa: F401 # type: ignore[import-unused]

from deesseia.eda.insights import Insights


class TestInsights:
    """Test automated insights functionality."""

    def test_generate_default(self, sample_dataframe: pd.DataFrame) -> None:
        """Test insights generation on default columns."""

        insights: Insights = Insights()
        result: dict[str, Any] = insights.generate(sample_dataframe)

        assert "skewed_columns" in result
        assert "outlier_columns" in result
        assert "high_correlation_pairs" in result
        assert "missing_analysis" in result
        assert "recommendations" in result

    def test_generate_no_numeric(
        self, sample_dataframe_non_numeric: pd.DataFrame
    ) -> None:
        """Test insights with no numeric columns."""

        insights: Insights = Insights()
        result: dict[str, Any] = insights.generate(sample_dataframe_non_numeric)

        assert result["skewed_columns"] == []
        assert result["outlier_columns"] == []
        assert result["high_correlation_pairs"] == []
        assert result["missing_analysis"]["has_missing"] is False
        assert "recommendations" in result

    def test_skew_detection(self, sample_dataframe_for_skew: pd.DataFrame) -> None:
        """Test skew detection."""

        insights: Insights = Insights()
        result: dict[str, Any] = insights.generate(sample_dataframe_for_skew)

        skewed_columns: list[dict[str, Any]] = result["skewed_columns"]
        skewed_names: list[str] = [s["column"] for s in skewed_columns]

        assert "right_skew" in skewed_names or "left_skew" in skewed_names

    def test_skew_detection_empty_series(
        self, sample_dataframe_mixed_empty_series: pd.DataFrame
    ) -> None:
        """Test skew detection with an empty series (all NaN)."""

        insights: Insights = Insights()
        result: dict[str, Any] = insights.generate(sample_dataframe_mixed_empty_series)

        skewed: list[dict[str, Any]] = result["skewed_columns"]

        assert len(skewed) == 0

    def test_outlier_detection_empty_series(
        self, sample_dataframe_mixed_empty_series: pd.DataFrame
    ) -> None:
        """Test outlier detection with an empty series (all NaN)."""

        insights: Insights = Insights()
        result: dict[str, Any] = insights.generate(sample_dataframe_mixed_empty_series)

        outliers: list[dict[str, Any]] = result["outlier_columns"]

        assert len(outliers) == 0

    def test_outlier_detection_iqr(
        self, sample_dataframe_for_outliers: pd.DataFrame
    ) -> None:
        """Test outlier detection using IQR method."""

        insights: Insights = Insights()
        result: dict[str, Any] = insights.generate(sample_dataframe_for_outliers)

        outlier_columns: list[dict[str, Any]] = result["outlier_columns"]
        outlier_names: list[str] = [o["column"] for o in outlier_columns]

        assert "with_outliers" in outlier_names

    def test_high_correlation_detection(
        self, sample_dataframe_for_correlation: pd.DataFrame
    ) -> None:
        """Test high correlation detection."""

        insights: Insights = Insights()
        result: dict[str, Any] = insights.generate(sample_dataframe_for_correlation)

        high_corr: list[dict[str, Any]] = result["high_correlation_pairs"]

        assert len(high_corr) > 0
        assert high_corr[0]["col1"] == "a" or high_corr[0]["col2"] == "a"

    def test_high_correlation_single_column(
        self, sample_dataframe_unique_values: pd.DataFrame
    ) -> None:
        """Test high correlation detection with a single column."""

        insights: Insights = Insights()
        result: dict[str, Any] = insights.generate(sample_dataframe_unique_values)

        assert result["high_correlation_pairs"] == []

    def test_missing_analysis(
        self, sample_dataframe_with_missing: pd.DataFrame
    ) -> None:
        """Test missing value analysis."""

        insights: Insights = Insights()
        result: dict[str, Any] = insights.generate(sample_dataframe_with_missing)

        missing: dict[str, Any] = result["missing_analysis"]

        assert missing["has_missing"] is True
        assert missing["total_missing"] == 3
        assert "col1" in missing["columns"]
        assert "col2" in missing["columns"]
        assert "col3" in missing["columns"]

    def test_missing_analysis_no_missing(self, sample_dataframe: pd.DataFrame) -> None:
        """Test missing value analysis with no missing values."""

        insights: Insights = Insights()
        result: dict[str, Any] = insights.generate(sample_dataframe)

        missing: dict[str, Any] = result["missing_analysis"]

        assert missing["has_missing"] is False
        assert missing["total_missing"] == 0
        assert missing["columns"] == {}

    def test_recommendations_generated(
        self, sample_dataframe_with_missing: pd.DataFrame
    ) -> None:
        """Test recommendations are generated."""

        insights: Insights = Insights()
        result: dict[str, Any] = insights.generate(sample_dataframe_with_missing)

        recommendations: list[str] = result["recommendations"]

        assert len(recommendations) > 0
        assert any("Missing values detected" in rec for rec in recommendations)

    def test_recommendations_many_high_correlations(
        self, sample_dataframe_many_correlations: pd.DataFrame
    ) -> None:
        """Test recommendations with many high correlations."""

        insights: Insights = Insights()
        result: dict[str, Any] = insights.generate(sample_dataframe_many_correlations)

        recommendations: list[str] = result["recommendations"]

        assert any("more pair" in rec for rec in recommendations)

    def test_recommendations_no_issues(
        self, sample_dataframe_clean: pd.DataFrame
    ) -> None:
        """Test recommendations when no issues are detected."""

        insights: Insights = Insights()
        result: dict[str, Any] = insights.generate(sample_dataframe_clean)

        recommendations: list[str] = result["recommendations"]

        assert any("No significant issues detected" in rec for rec in recommendations)

    def test_insights_with_columns_param(self, sample_dataframe: pd.DataFrame) -> None:
        """Test insights with specific columns."""

        insights: Insights = Insights()
        result: dict[str, Any] = insights.generate(
            sample_dataframe, columns=["age", "salary"]
        )

        assert "skewed_columns" in result
        assert "outlier_columns" in result
        assert "high_correlation_pairs" in result
