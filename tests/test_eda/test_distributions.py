from typing import Any

import pandas as pd
import pytest  # noqa: F401 # type: ignore[import-unused]

from deesseia.eda.distributions import Distributions


class TestDistributions:
    """Test distribution analysis functionality."""

    def test_histogram_default(self, sample_dataframe: pd.DataFrame) -> None:
        """Test histogram on a column."""

        dist: Distributions = Distributions()
        result: dict[str, Any] = dist.histogram(sample_dataframe, column="age")

        assert result["column"] == "age"
        assert result["count"] == 5
        assert "hist" in result
        assert "bin_edges" in result
        assert result["bins"] == 10

    def test_histogram_nonexistent_column(self, sample_dataframe: pd.DataFrame) -> None:
        """Test histogram with a nonexistent column."""

        dist: Distributions = Distributions()
        result: dict[str, Any] = dist.histogram(sample_dataframe, column="nonexistent")

        assert result == {}

    def test_histogram_empty_series(
        self, sample_dataframe_empty_series: pd.DataFrame
    ) -> None:
        """Test histogram with an empty series (all NaN)."""

        dist: Distributions = Distributions()
        result: dict[str, Any] = dist.histogram(
            sample_dataframe_empty_series, column="a"
        )

        assert result == {}

    def test_kde_default(self, sample_dataframe: pd.DataFrame) -> None:
        """Test KDE on a column."""

        dist: Distributions = Distributions()
        result: dict[str, Any] = dist.kde(sample_dataframe, column="age")

        assert result["column"] == "age"
        assert "x" in result
        assert "y" in result
        assert len(result["x"]) == 100
        assert len(result["y"]) == 100
        assert "bandwidth" in result

    def test_kde_nonexistent_column(self, sample_dataframe: pd.DataFrame) -> None:
        """Test KDE with a nonexistent column."""

        dist: Distributions = Distributions()
        result: dict[str, Any] = dist.kde(sample_dataframe, column="nonexistent")

        assert result == {}

    def test_kde_empty_series(
        self, sample_dataframe_empty_series: pd.DataFrame
    ) -> None:
        """Test KDE with an empty series (all NaN)."""

        dist: Distributions = Distributions()
        result: dict[str, Any] = dist.kde(sample_dataframe_empty_series, column="a")

        assert result == {}

    def test_kde_custom_bandwidth(self, sample_dataframe: pd.DataFrame) -> None:
        """Test KDE with custom bandwidth."""

        dist: Distributions = Distributions()
        result: dict[str, Any] = dist.kde(sample_dataframe, column="age", bandwidth=0.5)

        assert result["bandwidth"] == 0.5

    def test_boxplot_default(self, sample_dataframe: pd.DataFrame) -> None:
        """Test boxplot on a column."""

        dist: Distributions = Distributions()
        result: dict[str, Any] = dist.boxplot(sample_dataframe, column="age")

        assert result["column"] == "age"
        assert "min" in result
        assert "q1" in result
        assert "median" in result
        assert "q3" in result
        assert "max" in result
        assert "iqr" in result
        assert "lower_whisker" in result
        assert "upper_whisker" in result
        assert "outliers" in result
        assert result["count"] == 5

    def test_boxplot_nonexistent_column(self, sample_dataframe: pd.DataFrame) -> None:
        """Test boxplot with a nonexistent column."""

        dist: Distributions = Distributions()
        result: dict[str, Any] = dist.boxplot(sample_dataframe, column="nonexistent")

        assert result == {}

    def test_boxplot_empty_series(
        self, sample_dataframe_empty_series: pd.DataFrame
    ) -> None:
        """Test boxplot with an empty series (all NaN)."""

        dist: Distributions = Distributions()
        result: dict[str, Any] = dist.boxplot(sample_dataframe_empty_series, column="a")

        assert result == {}

    def test_violinplot_default(self, sample_dataframe: pd.DataFrame) -> None:
        """Test violin plot on a column."""

        dist: Distributions = Distributions()
        result: dict[str, Any] = dist.violinplot(sample_dataframe, column="age")

        assert result["column"] == "age"
        assert "x" in result
        assert "y" in result
        assert "min" in result
        assert "median" in result
        assert "max" in result
        assert result["count"] == 5

    def test_violinplot_nonexistent_column(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test violin plot with a nonexistent column."""

        dist: Distributions = Distributions()
        result: dict[str, Any] = dist.violinplot(sample_dataframe, column="nonexistent")

        assert result == {}

    def test_violinplot_empty_series(
        self, sample_dataframe_empty_series: pd.DataFrame
    ) -> None:
        """Test violin plot with an empty series (all NaN)."""

        dist: Distributions = Distributions()
        result: dict[str, Any] = dist.violinplot(
            sample_dataframe_empty_series, column="a"
        )

        assert result == {}

    def test_summary_default(self, sample_dataframe: pd.DataFrame) -> None:
        """Test complete distribution summary."""

        dist: Distributions = Distributions()
        result: dict[str, Any] = dist.summary(sample_dataframe, column="age")

        assert result["column"] == "age"
        assert result["count"] == 5
        assert "min" in result
        assert "max" in result
        assert "mean" in result
        assert "median" in result
        assert "std" in result
        assert "skew" in result
        assert "kurtosis" in result
        assert "histogram" in result
        assert "kde" in result
        assert "boxplot" in result

    def test_summary_nonexistent_column(self, sample_dataframe: pd.DataFrame) -> None:
        """Test summary with a nonexistent column."""

        dist: Distributions = Distributions()
        result: dict[str, Any] = dist.summary(sample_dataframe, column="nonexistent")

        assert result == {}

    def test_summary_empty_series(
        self, sample_dataframe_empty_series: pd.DataFrame
    ) -> None:
        """Test summary with an empty series (all NaN)."""

        dist: Distributions = Distributions()
        result = dist.summary(sample_dataframe_empty_series, column="a")

        assert result == {}
