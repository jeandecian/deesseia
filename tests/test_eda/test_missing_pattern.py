from typing import Any

import pandas as pd
import pytest

from deesseia.eda.missing_pattern import MissingPattern


class TestMissingPattern:
    """Test missing value pattern analysis functionality."""

    def test_summary_no_missing(self, sample_dataframe: pd.DataFrame) -> None:
        """Test missing summary with no missing values."""

        pattern: MissingPattern = MissingPattern()
        result: dict[str, Any] = pattern.summary(sample_dataframe)

        assert result["total_cells"] == 25
        assert result["total_missing"] == 0
        assert result["total_complete"] == 25
        assert result["missing_percentage"] == 0.0
        assert result["complete_rows"] == 5
        assert result["rows_with_missing"] == 0
        assert result["columns_with_missing"] == []
        assert result["total_columns_with_missing"] == 0

    def test_summary_with_missing(
        self, sample_dataframe_with_missing: pd.DataFrame
    ) -> None:
        """Test missing summary with missing values."""

        pattern: MissingPattern = MissingPattern()
        result: dict[str, Any] = pattern.summary(sample_dataframe_with_missing)

        assert result["total_cells"] == 20
        assert result["total_missing"] == 3
        assert result["total_complete"] == 17
        assert result["missing_percentage"] == 15.0
        assert "col1" in result["columns_with_missing"]
        assert "col2" in result["columns_with_missing"]
        assert "col3" in result["columns_with_missing"]

    def test_summary_empty_dataframe(self) -> None:
        """Test missing summary with empty DataFrame."""

        df: pd.DataFrame = pd.DataFrame()
        pattern: MissingPattern = MissingPattern()

        with pytest.raises(ValueError, match="DataFrame cannot be empty."):
            pattern.summary(df)

    def test_pattern_matrix_no_missing(self, sample_dataframe: pd.DataFrame) -> None:
        """Test pattern matrix with no missing values."""

        pattern: MissingPattern = MissingPattern()
        result: pd.DataFrame = pattern.pattern_matrix(sample_dataframe)

        assert not result.empty
        assert "pattern" in result.columns
        assert "count" in result.columns
        # All patterns should be "00000" (no missing)
        assert all(result["pattern"] == "0" * len(sample_dataframe.columns))

    def test_pattern_matrix_with_missing(
        self, sample_dataframe_with_missing: pd.DataFrame
    ) -> None:
        """Test pattern matrix with missing values."""

        pattern: MissingPattern = MissingPattern()
        result: pd.DataFrame = pattern.pattern_matrix(sample_dataframe_with_missing)

        assert not result.empty
        assert "pattern" in result.columns
        assert "count" in result.columns
        # Should have multiple patterns

    def test_pattern_matrix_empty_dataframe(self) -> None:
        """Test pattern matrix with empty DataFrame."""

        df: pd.DataFrame = pd.DataFrame()
        pattern: MissingPattern = MissingPattern()

        with pytest.raises(ValueError, match="DataFrame cannot be empty."):
            pattern.pattern_matrix(df)

    def test_missing_correlation_no_missing(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test missing correlation with no missing values."""

        pattern: MissingPattern = MissingPattern()
        result: pd.DataFrame = pattern.missing_correlation(sample_dataframe)

        assert not result.empty
        # All correlations should be NaN or 0 since no missing values
        assert result.isna().all().all() or (result == 0).all().all()

    def test_missing_correlation_with_missing(
        self, sample_dataframe_with_missing: pd.DataFrame
    ) -> None:
        """Test missing correlation with missing values."""

        pattern: MissingPattern = MissingPattern()
        result: pd.DataFrame = pattern.missing_correlation(
            sample_dataframe_with_missing
        )

        assert not result.empty
        # Should have the same columns as the input
        assert all(
            col in result.columns for col in sample_dataframe_with_missing.columns
        )

    def test_missing_correlation_empty_dataframe(self) -> None:
        """Test missing correlation with empty DataFrame."""

        df: pd.DataFrame = pd.DataFrame()
        pattern: MissingPattern = MissingPattern()

        with pytest.raises(ValueError, match="DataFrame cannot be empty."):
            pattern.missing_correlation(df)

    def test_missing_by_row_no_missing(self, sample_dataframe: pd.DataFrame) -> None:
        """Test missing by row with no missing values."""

        pattern: MissingPattern = MissingPattern()
        result: pd.Series = pattern.missing_by_row(sample_dataframe)

        assert len(result) == 5
        assert (result == 0).all()

    def test_missing_by_row_with_missing(
        self, sample_dataframe_with_missing: pd.DataFrame
    ) -> None:
        """Test missing by row with missing values."""

        pattern: MissingPattern = MissingPattern()
        result: pd.Series = pattern.missing_by_row(sample_dataframe_with_missing)

        assert len(result) == 5
        assert result.sum() == 3

    def test_missing_by_row_empty_dataframe(self) -> None:
        """Test missing by row with empty DataFrame."""

        df: pd.DataFrame = pd.DataFrame()
        pattern: MissingPattern = MissingPattern()

        with pytest.raises(ValueError, match="DataFrame cannot be empty."):
            pattern.missing_by_row(df)

    def test_detect_pattern_type_no_missing(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test pattern type detection with no missing values."""

        pattern: MissingPattern = MissingPattern()
        result: dict[str, Any] = pattern.detect_pattern_type(sample_dataframe)

        assert result["pattern_type"] == "No missing values"

    def test_detect_pattern_type_with_missing(
        self, sample_dataframe_with_missing: pd.DataFrame
    ) -> None:
        """Test pattern type detection with missing values."""

        pattern: MissingPattern = MissingPattern()
        result: dict[str, Any] = pattern.detect_pattern_type(
            sample_dataframe_with_missing
        )

        assert "pattern_type" in result
        assert "evidence" in result
        assert "correlations_found" in result

    def test_detect_pattern_type_mcar_low_variance(
        self, sample_dataframe_mcar_low_variance: pd.DataFrame
    ) -> None:
        """Test pattern type detection for MCAR with low variance in missing rows."""

        pattern: MissingPattern = MissingPattern()
        result: dict[str, Any] = pattern.detect_pattern_type(
            sample_dataframe_mcar_low_variance
        )

        assert "pattern_type" in result
        assert "MCAR" in result["pattern_type"]
        assert (
            "Missingness appears random" in result["evidence"]
            or "No strong correlations" in result["evidence"]
        )

    def test_detect_pattern_type_mcar_weak_evidence(
        self, sample_dataframe_mcar_weak_evidence: pd.DataFrame
    ) -> None:
        """Test pattern type detection for MCAR with weak evidence (higher variance)."""

        pattern: MissingPattern = MissingPattern()
        result: dict[str, Any] = pattern.detect_pattern_type(
            sample_dataframe_mcar_weak_evidence
        )

        assert "pattern_type" in result
        assert "MCAR" in result["pattern_type"]
        assert "evidence" in result
        assert len(result["evidence"]) > 0

    def test_detect_pattern_type_empty_dataframe(self) -> None:
        """Test pattern type detection with empty DataFrame."""

        df: pd.DataFrame = pd.DataFrame()
        pattern: MissingPattern = MissingPattern()

        with pytest.raises(ValueError, match="DataFrame cannot be empty."):
            pattern.detect_pattern_type(df)

    def test_detect_pattern_type_single_complete_case(
        self, sample_dataframe_single_complete_case: pd.DataFrame
    ) -> None:
        """Test pattern type detection when columns have only 1 complete case."""

        pattern: MissingPattern = MissingPattern()
        result: dict[str, Any] = pattern.detect_pattern_type(
            sample_dataframe_single_complete_case
        )

        assert "pattern_type" in result
        assert (
            result["pattern_type"]
            == "MCAR (Missing Completely At Random) - weak evidence"
        )
        assert "evidence" in result
        assert len(result["evidence"]) > 0
        assert result["correlations_found"] == []

    def test_report_no_missing(self, sample_dataframe: pd.DataFrame) -> None:
        """Test comprehensive report with no missing values."""

        pattern: MissingPattern = MissingPattern()
        result: dict[str, Any] = pattern.report(sample_dataframe)

        assert "summary" in result
        assert "pattern_matrix" in result
        assert "missing_correlation" in result
        assert "pattern_type" in result
        assert "top_missing_rows" in result
        assert "top_missing_columns" in result
        assert "has_missing" in result
        assert bool(result["has_missing"]) is False

    def test_report_with_missing(
        self, sample_dataframe_with_missing: pd.DataFrame
    ) -> None:
        """Test comprehensive report with missing values."""

        pattern: MissingPattern = MissingPattern()
        result: dict[str, Any] = pattern.report(sample_dataframe_with_missing)

        assert "summary" in result
        assert "pattern_matrix" in result
        assert "missing_correlation" in result
        assert "pattern_type" in result
        assert "top_missing_rows" in result
        assert "top_missing_columns" in result
        assert "has_missing" in result
        assert bool(result["has_missing"]) is True

    def test_report_empty_dataframe(self) -> None:
        """Test comprehensive report with empty DataFrame."""

        df: pd.DataFrame = pd.DataFrame()
        pattern: MissingPattern = MissingPattern()

        with pytest.raises(ValueError, match="DataFrame cannot be empty."):
            pattern.report(df)
