from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd


class FeatureCreator:
    """Create new features from existing data."""

    def __init__(self) -> None:
        """Initialize the FeatureCreator."""

        self._fitted: bool = False
        self._params: dict[str, Any] = {}

    def polynomial_features(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        degree: int = 2,
        include_bias: bool = False,
    ) -> pd.DataFrame:
        """Create polynomial features up to the specified degree."""

        result: pd.DataFrame = df.copy()
        cols: list[str] = (
            columns or df.select_dtypes(include=["number"]).columns.tolist()
        )

        if not cols:
            return result

        for col in cols:
            if col not in df.columns:
                continue

            for d in range(2, degree + 1):
                result[f"{col}^{d}"] = df[col] ** d

        if include_bias:
            result["bias"] = 1.0

        return result

    def interaction_features(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Create interaction terms between features."""

        result: pd.DataFrame = df.copy()

        if columns is None:
            cols: list[str] = df.select_dtypes(include=["number"]).columns.tolist()
        else:
            cols = [col for col in columns if col in df.columns]

        if len(cols) < 2:
            return result

        for col1, col2 in combinations(cols, 2):
            result[f"{col1}_{col2}"] = df[col1] * df[col2]

        return result

    def polynomial_and_interaction_features(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        degree: int = 2,
        include_bias: bool = False,
    ) -> pd.DataFrame:
        """Create both polynomial and interaction features."""

        result: pd.DataFrame = df.copy()
        cols: list[str] = (
            columns or df.select_dtypes(include=["number"]).columns.tolist()
        )

        if not cols:
            return result

        valid_cols: list[str] = [col for col in cols if col in df.columns]

        for col in valid_cols:
            for d in range(2, degree + 1):
                result[f"{col}^{d}"] = df[col] ** d

        if len(valid_cols) >= 2:
            for col1, col2 in combinations(valid_cols, 2):
                result[f"{col1}_{col2}"] = df[col1] * df[col2]

        if include_bias:
            result["bias"] = 1.0

        return result
