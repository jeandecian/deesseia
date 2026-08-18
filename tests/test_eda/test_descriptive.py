from typing import Any

import pandas as pd
import pytest  # noqa: F401 # type: ignore[import-unused]

from deesseia.eda.descriptive import DescriptiveStats


class TestDescriptiveStats:
    """Test descriptive statistics functionality."""

    def test_summary_default(self, sample_dataframe: pd.DataFrame) -> None:
        """Test summary on all numeric columns."""

        stats: DescriptiveStats = DescriptiveStats()
        result: dict[str, Any] = stats.summary(sample_dataframe)

        assert "age" in result
        assert "salary" in result
        assert "id" in result
        assert "mean" in result["age"]
        assert "median" in result["age"]
        assert "std" in result["age"]
        assert result["age"]["count"] == 5
        assert result["age"]["missing_count"] == 0

    def test_summary_with_columns(self, sample_dataframe: pd.DataFrame) -> None:
        """Test summary on specific columns."""

        stats: DescriptiveStats = DescriptiveStats()
        result: dict[str, Any] = stats.summary(sample_dataframe, columns=["age"])

        assert "age" in result
        assert "salary" not in result
        assert "id" not in result

    def test_summary_with_missing(
        self, sample_dataframe_with_missing: pd.DataFrame
    ) -> None:
        """Test summary with missing values."""

        stats: DescriptiveStats = DescriptiveStats()
        result: dict[str, Any] = stats.summary(sample_dataframe_with_missing)

        assert "col1" in result
        assert result["col1"]["missing_count"] == 1
        assert result["col1"]["missing_percentage"] == 20.0

    def test_summary_no_numeric(
        self, sample_dataframe_non_numeric: pd.DataFrame
    ) -> None:
        """Test summary with no numeric columns."""

        stats: DescriptiveStats = DescriptiveStats()
        result: dict[str, Any] = stats.summary(sample_dataframe_non_numeric)

        assert result == {}

    def test_summary_nonexistent_column(self, sample_dataframe: pd.DataFrame) -> None:
        """Test summary with a nonexistent column."""

        stats: DescriptiveStats = DescriptiveStats()
        result: dict[str, Any] = stats.summary(
            sample_dataframe, columns=["age", "nonexistent"]
        )

        assert "age" in result
        assert "nonexistent" not in result

    def test_summary_non_numeric_column(self, sample_dataframe: pd.DataFrame) -> None:
        """Test summary with non-numeric column specified (should be ignored)."""

        stats: DescriptiveStats = DescriptiveStats()
        result: dict[str, Any] = stats.summary(sample_dataframe, columns=["category"])

        assert result == {}

    def test_summary_empty_series(
        self, sample_dataframe_empty_series: pd.DataFrame
    ) -> None:
        """Test summary with an empty series (all NaN)."""

        stats: DescriptiveStats = DescriptiveStats()
        result: dict[str, Any] = stats.summary(sample_dataframe_empty_series)

        assert result == {}

    def test_summary_mode_with_bimodal(
        self, sample_dataframe_bimodal: pd.DataFrame
    ) -> None:
        """Test summary with bimodal distribution."""

        stats: DescriptiveStats = DescriptiveStats()
        result: dict[str, Any] = stats.summary(sample_dataframe_bimodal)

        assert "a" in result
        assert sorted(result["a"]["mode"]) == [1, 2]

    def test_five_number_summary_default(self, sample_dataframe: pd.DataFrame) -> None:
        """Test five-number summary on all numeric columns."""

        stats: DescriptiveStats = DescriptiveStats()
        result: dict[str, Any] = stats.five_number_summary(sample_dataframe)

        assert "age" in result
        assert "salary" in result
        assert "id" in result
        assert "min" in result["age"]
        assert "q1" in result["age"]
        assert "median" in result["age"]
        assert "q3" in result["age"]
        assert "max" in result["age"]

    def test_five_number_summary_with_columns(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test five-number summary on specific columns."""

        stats: DescriptiveStats = DescriptiveStats()
        result: dict[str, Any] = stats.five_number_summary(
            sample_dataframe, columns=["age"]
        )

        assert "age" in result
        assert "salary" not in result
        assert "id" not in result

    def test_five_number_summary_no_numeric(
        self, sample_dataframe_non_numeric: pd.DataFrame
    ) -> None:
        """Test five-number summary with no numeric columns."""

        stats: DescriptiveStats = DescriptiveStats()
        result: dict[str, Any] = stats.five_number_summary(sample_dataframe_non_numeric)

        assert result == {}

    def test_five_number_summary_with_missing(
        self, sample_dataframe_with_missing: pd.DataFrame
    ) -> None:
        """Test five-number summary with missing values."""

        stats: DescriptiveStats = DescriptiveStats()
        result: dict[str, Any] = stats.five_number_summary(
            sample_dataframe_with_missing
        )

        assert "col1" in result
        assert "col2" in result
        assert "col4" in result
        assert "col3" not in result

    def test_five_number_summary_empty_series(
        self, sample_dataframe_empty_series: pd.DataFrame
    ) -> None:
        """Test five-number summary with an empty series (all NaN)."""

        stats: DescriptiveStats = DescriptiveStats()
        result: dict[str, Any] = stats.five_number_summary(
            sample_dataframe_empty_series
        )

        assert result == {}

    def test_five_number_summary_non_numeric_column(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test five-number summary with non-numeric column specified."""

        stats: DescriptiveStats = DescriptiveStats()
        result: dict[str, Any] = stats.five_number_summary(
            sample_dataframe, columns=["category"]
        )

        assert result == {}

    def test_five_number_summary_nonexistent_column(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test five-number summary with a nonexistent column."""

        stats: DescriptiveStats = DescriptiveStats()
        result: dict[str, Any] = stats.five_number_summary(
            sample_dataframe, columns=["age", "nonexistent"]
        )

        assert "age" in result
        assert "nonexistent" not in result
