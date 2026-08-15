import numpy as np
import pandas as pd
import pytest  # noqa: F401 # type: ignore[import-unused]

from deesseia.preprocess.scaler import Scaler


class TestScaler:
    """Test feature scaling functionality."""

    def test_minmax_scale_default(self, sample_dataframe_for_scaling: pd.DataFrame):
        """Test MinMax scaling to [0, 1] range."""

        scaler: Scaler = Scaler()
        result: pd.DataFrame = scaler.minmax_scale(sample_dataframe_for_scaling)

        assert result["a"].min() == 0.0
        assert result["a"].max() == 1.0
        assert result["b"].min() == 0.0
        assert result["b"].max() == 1.0
        assert result["c"].min() == 0.0
        assert result["c"].max() == 1.0

    def test_minmax_scale_custom_range(
        self, sample_dataframe_for_scaling: pd.DataFrame
    ):
        """Test MinMax scaling to custom range."""

        scaler: Scaler = Scaler()
        result: pd.DataFrame = scaler.minmax_scale(
            sample_dataframe_for_scaling[["a"]], feature_range=(-1, 1)
        )

        assert result["a"].min() == -1.0
        assert result["a"].max() == 1.0

    def test_minmax_scale_single_value(
        self, sample_dataframe_single_value_column: pd.DataFrame
    ):
        """Test MinMax scaling with a column of identical values."""

        scaler: Scaler = Scaler()
        result: pd.DataFrame = scaler.minmax_scale(sample_dataframe_single_value_column)

        assert (result["a"] == 0.0).all()

    def test_standard_scale(self, sample_dataframe_for_scaling: pd.DataFrame):
        """Test Standard scaling (zero mean, unit variance)."""

        scaler: Scaler = Scaler()
        result: pd.DataFrame = scaler.standard_scale(sample_dataframe_for_scaling)

        assert abs(result["a"].mean()) < 1e-10
        assert abs(result["a"].std() - 1.0) < 1e-10
        assert abs(result["b"].mean()) < 1e-10
        assert abs(result["b"].std() - 1.0) < 1e-10

    def test_standard_scale_single_value(
        self, sample_dataframe_single_value_column: pd.DataFrame
    ):
        """Test Standard scaling with a column of identical values."""

        scaler: Scaler = Scaler()
        result: pd.DataFrame = scaler.standard_scale(
            sample_dataframe_single_value_column
        )

        assert (result["a"] == 0.0).all()
        assert abs(result["b"].mean()) < 1e-10

    def test_robust_scale(self, sample_dataframe_with_outliers: pd.DataFrame):
        """Test Robust scaling (median and IQR)."""

        scaler: Scaler = Scaler()
        result: pd.DataFrame = scaler.robust_scale(sample_dataframe_with_outliers)

        # Median should be 0
        assert abs(result["a"].median()) < 1e-10
        assert abs(result["b"].median()) < 1e-10
        assert abs(result["c"].median()) < 1e-10

    def test_robust_scale_zero_iqr(self, sample_dataframe_zero_iqr: pd.DataFrame):
        """Test Robust scaling with a column where IQR = 0."""

        scaler: Scaler = Scaler()
        result: pd.DataFrame = scaler.robust_scale(sample_dataframe_zero_iqr)

        assert (result["a"] == 0.0).all()
        assert abs(result["b"].median()) < 1e-10

    def test_maxabs_scale(self, sample_dataframe_for_scaling: pd.DataFrame):
        """Test MaxAbs scaling."""

        scaler: Scaler = Scaler()
        result: pd.DataFrame = scaler.maxabs_scale(sample_dataframe_for_scaling)

        assert result["a"].min() == 0.2
        assert result["a"].max() == 1.0
        assert result["b"].min() == 0.2
        assert result["b"].max() == 1.0
        assert result["d"].min() == -1.0
        assert result["d"].max() == 1.0

    def test_maxabs_scale_all_zeros(self, sample_dataframe_all_zeros: pd.DataFrame):
        """Test MaxAbs scaling with all zeros."""

        scaler: Scaler = Scaler()
        result: pd.DataFrame = scaler.maxabs_scale(sample_dataframe_all_zeros)

        assert (result["a"] == 0.0).all()

    def test_log_transform(self, sample_dataframe_for_log_transform: pd.DataFrame):
        """Test log transformation."""

        scaler: Scaler = Scaler()
        result: pd.DataFrame = scaler.log_transform(sample_dataframe_for_log_transform)

        # Log of 1 is 0
        assert result["a"].iloc[0] == 0.0

    def test_log_transform_with_zeros(self, sample_dataframe_with_zeros: pd.DataFrame):
        """Test log transformation with zeros (should add epsilon)."""

        scaler: Scaler = Scaler()
        result: pd.DataFrame = scaler.log_transform(sample_dataframe_with_zeros)

        assert not np.isinf(result["a"]).any()

    def test_log_transform_custom_base(
        self, sample_dataframe_for_log_transform: pd.DataFrame
    ):
        """Test log transformation with custom base."""

        scaler: Scaler = Scaler()
        result: pd.DataFrame = scaler.log_transform(
            sample_dataframe_for_log_transform, base=10
        )

        # Log10 of 10 is 1
        assert result["a"].iloc[1] == 1.0

    def test_scaler_with_column_selection(
        self, sample_dataframe_for_scaling: pd.DataFrame
    ):
        """Test scaling only selected columns."""

        scaler: Scaler = Scaler()
        result: pd.DataFrame = scaler.minmax_scale(
            sample_dataframe_for_scaling, columns=["a", "b"]
        )

        # Columns c and d should be unchanged
        assert result["a"].min() == 0.0
        assert result["b"].min() == 0.0
        assert result["c"].iloc[0] == 100
        assert result["d"].iloc[0] == -5

    def test_nonexistent_column_all_methods(
        self, sample_dataframe_for_scaling: pd.DataFrame
    ):
        """Test that all scaling methods handle nonexistent columns gracefully."""

        scaler: Scaler = Scaler()
        df: pd.DataFrame = sample_dataframe_for_scaling.copy()
        columns: list[str] = ["a", "nonexistent", "b"]

        result1: pd.DataFrame = scaler.minmax_scale(df, columns=columns)
        result2: pd.DataFrame = scaler.standard_scale(df, columns=columns)
        result3: pd.DataFrame = scaler.robust_scale(df, columns=columns)
        result4: pd.DataFrame = scaler.maxabs_scale(df, columns=columns)
        result5: pd.DataFrame = scaler.log_transform(df, columns=columns)

        assert result1["a"].min() == 0.0
        assert result1["b"].min() == 0.0
        assert abs(result2["a"].mean()) < 1e-10
        assert abs(result2["b"].mean()) < 1e-10
        assert abs(result3["a"].median()) < 1e-10
        assert abs(result3["b"].median()) < 1e-10
        assert result4["a"].min() == 0.2
        assert result4["b"].min() == 0.2
        assert result5["a"].iloc[0] == 0.0

    def test_scaler_nonexistent_column(
        self, sample_dataframe_for_scaling: pd.DataFrame
    ):
        """Test scaling with a nonexistent column (should be skipped)."""

        scaler: Scaler = Scaler()
        result: pd.DataFrame = scaler.minmax_scale(
            sample_dataframe_for_scaling, columns=["a", "nonexistent"]
        )

        assert result["a"].min() == 0.0
        assert "nonexistent" not in result.columns
