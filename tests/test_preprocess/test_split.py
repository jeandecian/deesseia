from typing import cast

import numpy as np
import pandas as pd
import pytest

from deesseia.preprocess.split import Splitter, TrainValTestSplitResult


class TestSplitter:
    """Test data splitting functionality."""

    def test_train_test_split_X_only(
        self, sample_dataframe_for_split: pd.DataFrame
    ) -> None:
        """Test train_test_split with features only."""

        splitter: Splitter = Splitter()
        result: tuple[np.ndarray, np.ndarray] = cast(
            tuple[np.ndarray, np.ndarray],
            splitter.train_test_split(
                sample_dataframe_for_split,
                test_size=0.2,
                random_state=42,
            ),
        )

        assert len(result) == 2
        assert len(result[0]) == 8
        assert len(result[1]) == 2

    def test_train_test_split_X_y(
        self,
        sample_dataframe_for_split: pd.DataFrame,
        sample_target_for_split: pd.Series,
    ) -> None:
        """Test train_test_split with features and target."""

        splitter: Splitter = Splitter()
        result: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] = cast(
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            splitter.train_test_split(
                sample_dataframe_for_split,
                sample_target_for_split,
                test_size=0.2,
                random_state=42,
            ),
        )

        assert len(result) == 4
        assert len(result[0]) == 8
        assert len(result[2]) == 8

    def test_train_test_split_stratify(
        self,
        sample_dataframe_for_split: pd.DataFrame,
        sample_target_binary_for_split: pd.Series,
    ) -> None:
        """Test train_test_split with stratification."""

        splitter: Splitter = Splitter()
        result: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] = cast(
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            splitter.train_test_split(
                sample_dataframe_for_split,
                sample_target_binary_for_split,
                test_size=0.2,
                random_state=42,
                stratify=sample_target_binary_for_split,
            ),
        )

        assert len(result) == 4

    def test_train_val_test_split_X_only(
        self, sample_dataframe_for_split: pd.DataFrame
    ) -> None:
        """Test train_val_test_split with features only."""

        splitter: Splitter = Splitter()
        result: TrainValTestSplitResult = splitter.train_val_test_split(
            sample_dataframe_for_split,
            train_size=0.6,
            val_size=0.2,
            test_size=0.2,
            random_state=42,
        )

        assert isinstance(result, TrainValTestSplitResult)
        assert result.y_train is None
        assert result.y_val is None
        assert result.y_test is None
        assert len(result.X_train) + len(result.x_val) + len(result.X_test) == 10
        assert len(result.X_train) > 0
        assert len(result.x_val) > 0
        assert len(result.X_test) > 0

    def test_train_val_test_split_X_y(
        self,
        sample_dataframe_for_split: pd.DataFrame,
        sample_target_for_split: pd.Series,
    ) -> None:
        """Test train_val_test_split with features and target."""

        splitter: Splitter = Splitter()
        result: TrainValTestSplitResult = splitter.train_val_test_split(
            sample_dataframe_for_split,
            sample_target_for_split,
            train_size=0.6,
            val_size=0.2,
            test_size=0.2,
            random_state=42,
        )

        assert isinstance(result, TrainValTestSplitResult)
        assert len(result.X_train) + len(result.x_val) + len(result.X_test) == 10
        assert len(result.X_train) > 0
        assert len(result.x_val) > 0
        assert len(result.X_test) > 0
        assert result.y_train is not None
        assert result.y_val is not None
        assert result.y_test is not None
        assert len(result.X_train) == len(result.y_train)
        assert len(result.x_val) == len(result.y_val)
        assert len(result.X_test) == len(result.y_test)

    def test_train_val_test_split_invalid_sizes(
        self, sample_dataframe_for_split: pd.DataFrame
    ) -> None:
        """Test train_val_test_split with invalid sizes."""

        splitter: Splitter = Splitter()

        with pytest.raises(
            ValueError, match="train_size \\+ val_size \\+ test_size must equal 1.0"
        ):
            splitter.train_val_test_split(
                sample_dataframe_for_split,
                train_size=0.5,
                val_size=0.3,
                test_size=0.3,
            )

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
