from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold as SklearnKFold
from sklearn.model_selection import StratifiedKFold as SklearnStratifiedKFold
from sklearn.model_selection import TimeSeriesSplit as SklearnTimeSeriesSplit


class Splitter:
    """Split data for model training and validation."""

    def __init__(self) -> None:
        """Initialize the Splitter."""

        self._fitted: bool = False
        self._params: dict[str, Any] = {}

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
