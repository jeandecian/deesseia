import numpy as np
import pandas as pd
import pytest  # noqa: F401 # type: ignore[import-unused]

from deesseia.preprocess.split import Splitter


class TestSplitter:
    """Test data splitting functionality."""

    def test_kfold(self, sample_dataframe_for_split: pd.DataFrame) -> None:
        """Test K-Fold cross-validation."""

        splitter: Splitter = Splitter()
        splits: list[tuple[np.ndarray, np.ndarray]] = splitter.kfold(
            sample_dataframe_for_split, n_splits=5, random_state=42
        )

        assert len(splits) == 5
        for train_idx, test_idx in splits:
            assert len(train_idx) == 8
            assert len(test_idx) == 2

    def test_kfold_with_y(
        self,
        sample_dataframe_for_split: pd.DataFrame,
        sample_target_for_split: pd.Series,
    ) -> None:
        """Test K-Fold cross-validation with target."""

        splitter: Splitter = Splitter()
        splits: list[tuple[np.ndarray, np.ndarray]] = splitter.kfold(
            sample_dataframe_for_split,
            sample_target_for_split,
            n_splits=5,
            random_state=42,
        )

        assert len(splits) == 5

    def test_stratified_kfold(
        self,
        sample_dataframe_for_split: pd.DataFrame,
        sample_target_binary_for_split: pd.Series,
    ) -> None:
        """Test Stratified K-Fold cross-validation."""

        splitter: Splitter = Splitter()
        splits: list[tuple[np.ndarray, np.ndarray]] = splitter.stratified_kfold(
            sample_dataframe_for_split,
            sample_target_binary_for_split,
            n_splits=5,
            random_state=42,
        )

        assert len(splits) == 5

    def test_timeseries_split(self, sample_dataframe_for_split: pd.DataFrame) -> None:
        """Test Time Series cross-validation."""

        splitter: Splitter = Splitter()
        splits: list[tuple[np.ndarray, np.ndarray]] = splitter.timeseries_split(
            sample_dataframe_for_split, n_splits=3
        )

        assert len(splits) == 3

    def test_timeseries_split_with_y(
        self,
        sample_dataframe_for_split: pd.DataFrame,
        sample_target_for_split: pd.Series,
    ) -> None:
        """Test Time Series cross-validation with target."""

        splitter: Splitter = Splitter()
        splits: list[tuple[np.ndarray, np.ndarray]] = splitter.timeseries_split(
            sample_dataframe_for_split,
            sample_target_for_split,
            n_splits=3,
        )

        assert len(splits) == 3

    def test_timeseries_split_with_max_train_size(
        self, sample_dataframe_for_split: pd.DataFrame
    ) -> None:
        """Test Time Series cross-validation with max_train_size."""

        splitter: Splitter = Splitter()
        splits: list[tuple[np.ndarray, np.ndarray]] = splitter.timeseries_split(
            sample_dataframe_for_split,
            n_splits=3,
            max_train_size=4,
        )

        assert len(splits) == 3
        for train_idx, _ in splits:
            assert len(train_idx) <= 4
