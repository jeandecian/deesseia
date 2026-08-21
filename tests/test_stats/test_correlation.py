from pathlib import Path
from typing import Any

import pandas as pd
import pytest  # noqa: F401 # type: ignore[import-unused]

from deesseia.stats.correlation import Correlation


class TestCorrelation:
    """Test correlation analysis functionality."""

    def test_correlation_matrix_default(self, sample_dataframe: pd.DataFrame) -> None:
        """Test correlation matrix on all numeric columns."""

        corr: Correlation = Correlation()
        result: pd.DataFrame = corr.correlation_matrix(sample_dataframe)

        assert isinstance(result, pd.DataFrame)
        assert "age" in result.columns
        assert "salary" in result.columns
        assert "id" in result.columns
        assert result.shape == (3, 3)

    def test_correlation_matrix_with_columns(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test correlation matrix on specific columns."""

        corr: Correlation = Correlation()
        result: pd.DataFrame = corr.correlation_matrix(
            sample_dataframe, columns=["age", "salary"]
        )

        assert result.shape == (2, 2)
        assert "age" in result.columns
        assert "salary" in result.columns
        assert "id" not in result.columns

    def test_correlation_matrix_no_numeric(
        self, sample_dataframe_non_numeric: pd.DataFrame
    ) -> None:
        """Test correlation matrix with no numeric columns."""

        corr: Correlation = Correlation()
        result: pd.DataFrame = corr.correlation_matrix(sample_dataframe_non_numeric)

        assert result.empty

    def test_correlation_matrix_methods(self, sample_dataframe: pd.DataFrame) -> None:
        """Test correlation matrix with different methods."""

        corr: Correlation = Correlation()
        result_pearson: pd.DataFrame = corr.correlation_matrix(
            sample_dataframe, method="pearson"
        )
        result_spearman: pd.DataFrame = corr.correlation_matrix(
            sample_dataframe, method="spearman"
        )
        result_kendall: pd.DataFrame = corr.correlation_matrix(
            sample_dataframe, method="kendall"
        )

        assert isinstance(result_pearson, pd.DataFrame)
        assert isinstance(result_spearman, pd.DataFrame)
        assert isinstance(result_kendall, pd.DataFrame)

    def test_high_correlations_default(
        self, sample_dataframe_high_correlation: pd.DataFrame
    ) -> None:
        """Test high correlations detection."""

        corr: Correlation = Correlation()
        result: list[dict[str, float | str]] = corr.high_correlations(
            sample_dataframe_high_correlation, threshold=0.8
        )

        assert len(result) > 0
        assert result[0]["col1"] == "a" or result[0]["col2"] == "a"

    def test_high_correlations_with_columns(
        self, sample_dataframe_high_correlation: pd.DataFrame
    ) -> None:
        """Test high correlations with specific columns."""

        corr: Correlation = Correlation()
        result: list[dict[str, float | str]] = corr.high_correlations(
            sample_dataframe_high_correlation,
            columns=["a", "c"],
            threshold=0.8,
        )

        assert len(result) > 0
        assert {"a", "c"} == {result[0]["col1"], result[0]["col2"]}

    def test_high_correlations_with_non_numeric_column(
        self, sample_dataframe_high_correlation: pd.DataFrame
    ) -> None:
        """Test high correlations with non-numeric column specified (should be ignored)."""

        corr: Correlation = Correlation()
        result: list[dict[str, float | str]] = corr.high_correlations(
            sample_dataframe_high_correlation,
            columns=["a", "nonexistent", "c"],
            threshold=0.8,
        )

        assert len(result) > 0
        assert {"a", "c"} == {result[0]["col1"], result[0]["col2"]}

    def test_high_correlations_no_columns(self, sample_dataframe: pd.DataFrame) -> None:
        """Test high correlations with no columns."""

        corr: Correlation = Correlation()
        result: list[dict[str, float | str]] = corr.high_correlations(
            sample_dataframe, threshold=0.9
        )

        assert isinstance(result, list)

    def test_high_correlations_single_column(
        self, sample_dataframe_single_column: pd.DataFrame
    ) -> None:
        """Test high correlations with a single column."""

        corr: Correlation = Correlation()
        result: list[dict[str, float | str]] = corr.high_correlations(
            sample_dataframe_single_column
        )

        assert result == []

    def test_plot_heatmap_show(self, sample_dataframe: pd.DataFrame) -> None:
        """Test heatmap plotting with show=True (default)."""

        corr: Correlation = Correlation()
        result: Any = corr.plot_heatmap(sample_dataframe, figsize=(4, 3), show=True)

        assert result is None

    def test_plot_heatmap_no_show(self, sample_dataframe: pd.DataFrame) -> None:
        """Test heatmap plotting with show=False (should not display)."""

        corr: Correlation = Correlation()
        result: Any = corr.plot_heatmap(sample_dataframe, figsize=(4, 3), show=False)

        assert result is None

    def test_plot_heatmap_save(
        self, sample_dataframe: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Test heatmap plotting with save_path."""

        import os

        save_path: Path = tmp_path / "heatmap.png"

        corr: Correlation = Correlation()
        result: Any = corr.plot_heatmap(
            sample_dataframe,
            figsize=(4, 3),
            show=False,
            save_path=str(save_path),
        )

        assert result is None
        assert os.path.exists(save_path)

    def test_plot_heatmap_return_fig(self, sample_dataframe: pd.DataFrame) -> None:
        """Test heatmap plotting with return_fig=True."""

        corr: Correlation = Correlation()
        result: Any = corr.plot_heatmap(
            sample_dataframe, figsize=(4, 3), show=False, return_fig=True
        )

        assert result is not None
        assert hasattr(result, "savefig")

    def test_plot_heatmap_save_and_return(
        self, sample_dataframe: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Test heatmap plotting with both save_path and return_fig."""

        import os

        save_path: Path = tmp_path / "heatmap_return.png"

        corr: Correlation = Correlation()
        result: Any = corr.plot_heatmap(
            sample_dataframe,
            figsize=(4, 3),
            show=False,
            save_path=str(save_path),
            return_fig=True,
        )

        assert result is not None
        assert hasattr(result, "savefig")
        assert os.path.exists(save_path)

    def test_plot_heatmap_no_numeric(
        self, sample_dataframe_non_numeric: pd.DataFrame
    ) -> None:
        """Test heatmap with no numeric columns."""

        corr: Correlation = Correlation()
        result: Any = corr.plot_heatmap(sample_dataframe_non_numeric, show=False)

        assert result is None

    def test_plot_heatmap_with_columns(self, sample_dataframe: pd.DataFrame) -> None:
        """Test heatmap with specific columns."""

        corr: Correlation = Correlation()
        result: Any = corr.plot_heatmap(
            sample_dataframe,
            columns=["age", "salary"],
            figsize=(4, 3),
            show=False,
        )

        assert result is None

    def test_plot_heatmap_with_method(self, sample_dataframe: pd.DataFrame) -> None:
        """Test heatmap with different correlation method."""

        corr: Correlation = Correlation()
        result: Any = corr.plot_heatmap(
            sample_dataframe,
            method="spearman",
            figsize=(4, 3),
            show=False,
        )

        assert result is None

    def test_plot_heatmap_with_custom_cmap(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test heatmap with custom colormap."""

        corr: Correlation = Correlation()
        result: Any = corr.plot_heatmap(
            sample_dataframe,
            figsize=(4, 3),
            cmap="coolwarm",
            show=False,
        )

        assert result is None

    def test_plot_heatmap_without_annot(self, sample_dataframe: pd.DataFrame) -> None:
        """Test heatmap without annotations."""

        corr: Correlation = Correlation()
        result: Any = corr.plot_heatmap(
            sample_dataframe,
            figsize=(4, 3),
            annot=False,
            show=False,
        )
        assert result is None
