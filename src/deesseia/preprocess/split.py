from __future__ import annotations

from typing import Any, NamedTuple, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold as SklearnKFold
from sklearn.model_selection import StratifiedKFold as SklearnStratifiedKFold
from sklearn.model_selection import TimeSeriesSplit as SklearnTimeSeriesSplit
from sklearn.model_selection import (
    train_test_split as sk_train_test_split,  # type: ignore
)


class TrainValTestSplitResult(NamedTuple):
    """Result of train_val_test_split."""

    X_train: np.ndarray | pd.DataFrame
    x_val: np.ndarray | pd.DataFrame
    X_test: np.ndarray | pd.DataFrame
    y_train: np.ndarray | pd.Series | None = None
    y_val: np.ndarray | pd.Series | None = None
    y_test: np.ndarray | pd.Series | None = None


class Splitter:
    """Split data for model training and validation."""

    def __init__(self) -> None:
        """Initialize the Splitter."""

        self._fitted: bool = False
        self._params: dict[str, Any] = {}

    def train_test_split(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        test_size: float = 0.2,
        train_size: float | None = None,
        random_state: int | None = None,
        shuffle: bool = True,
        stratify: pd.Series | np.ndarray | None = None,
    ) -> (
        tuple[np.ndarray, np.ndarray]
        | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ):
        """Split data into train and test sets.

        Returns:
            (X_train, X_test) or (X_train, X_test, y_train, y_test)
        """

        if y is None:
            return cast(
                tuple[np.ndarray, np.ndarray],
                sk_train_test_split(
                    X,
                    test_size=test_size,
                    train_size=train_size,
                    random_state=random_state,
                    shuffle=shuffle,
                    stratify=stratify,
                ),
            )

        return cast(
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            sk_train_test_split(
                X,
                y,
                test_size=test_size,
                train_size=train_size,
                random_state=random_state,
                shuffle=shuffle,
                stratify=stratify,
            ),
        )

    def train_val_test_split(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        train_size: float = 0.6,
        val_size: float = 0.2,
        test_size: float = 0.2,
        random_state: int | None = None,
        shuffle: bool = True,
        stratify: pd.Series | np.ndarray | None = None,
    ) -> TrainValTestSplitResult:
        """Split data into train, validation, and test sets.

        Returns:
            (X_train, x_val, X_test) or (X_train, x_val, X_test, y_train, y_val, y_test)
        """

        if abs((train_size + val_size + test_size) - 1.0) > 1e-9:
            raise ValueError("train_size + val_size + test_size must equal 1.0")

        test_ratio: float = test_size / (train_size + val_size)

        if y is None:
            x_train_val_test_split: tuple[np.ndarray, np.ndarray] = cast(
                tuple[np.ndarray, np.ndarray],
                sk_train_test_split(
                    X,
                    test_size=test_ratio,
                    random_state=random_state,
                    shuffle=shuffle,
                    stratify=stratify,
                ),
            )
            x_train_val: np.ndarray = x_train_val_test_split[0]
            X_test: np.ndarray = x_train_val_test_split[1]

            val_ratio: float = val_size / (train_size + val_size)
            x_train_val_split: tuple[np.ndarray, np.ndarray] = cast(
                tuple[np.ndarray, np.ndarray],
                sk_train_test_split(
                    x_train_val,
                    test_size=val_ratio,
                    random_state=random_state,
                    shuffle=shuffle,
                    stratify=stratify,
                ),
            )
            X_train: np.ndarray = x_train_val_split[0]
            x_val: np.ndarray = x_train_val_split[1]

            return TrainValTestSplitResult(
                X_train=X_train,
                x_val=x_val,
                X_test=X_test,
            )

        x_y_train_val_test_split: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ] = cast(
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            sk_train_test_split(
                X,
                y,
                test_size=test_ratio,
                random_state=random_state,
                shuffle=shuffle,
                stratify=stratify,
            ),
        )
        x_train_val = x_y_train_val_test_split[0]
        X_test = x_y_train_val_test_split[1]
        y_train_val: np.ndarray = x_y_train_val_test_split[2]
        y_test: np.ndarray = x_y_train_val_test_split[3]

        val_ratio = val_size / (train_size + val_size)
        x_y_train_val_split: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] = (
            cast(
                tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                sk_train_test_split(
                    x_train_val,
                    y_train_val,
                    test_size=val_ratio,
                    random_state=random_state,
                    shuffle=shuffle,
                    stratify=stratify,
                ),
            )
        )
        X_train = x_y_train_val_split[0]
        x_val = x_y_train_val_split[1]
        y_train: np.ndarray = x_y_train_val_split[2]
        y_val: np.ndarray = x_y_train_val_split[3]

        return TrainValTestSplitResult(
            X_train=X_train,
            x_val=x_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
        )

    def kfold(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate K-Fold cross-validation splits."""

        kf: SklearnKFold = SklearnKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

        if y is None:
            return list(kf.split(X))

        return list(kf.split(X, y))

    def stratified_kfold(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate Stratified K-Fold cross-validation splits."""

        skf: SklearnStratifiedKFold = SklearnStratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )

        return list(skf.split(X, y))

    def timeseries_split(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        n_splits: int = 5,
        max_train_size: int | None = None,
        test_size: int | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate Time Series cross-validation splits."""

        tscv: SklearnTimeSeriesSplit = SklearnTimeSeriesSplit(
            n_splits=n_splits,
            max_train_size=max_train_size,
            test_size=test_size,
        )

        if y is None:
            return list(tscv.split(X))

        return list(tscv.split(X, y))
