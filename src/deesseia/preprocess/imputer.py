from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression


class Imputer:
    """Handle missing values with various imputation strategies."""

    def __init__(self) -> None:
        """Initialize the Imputer."""

        self._fitted: bool = False
        self._params: dict[str, Any] = {}

    def mean_impute(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fill missing values with column mean.

        Args:
            df: DataFrame to impute.
            columns: List of columns to impute. If None, imputes all numeric columns.

        Returns:
            DataFrame with missing values filled.
        """

        result: pd.DataFrame = df.copy()
        cols: list[str] = (
            columns or df.select_dtypes(include=["number"]).columns.tolist()
        )

        for col in cols:
            if col not in df.columns:
                continue

            mean_val: float = df[col].mean()
            if pd.isna(mean_val):
                result[col] = df[col]
            else:
                result[col] = df[col].fillna(mean_val)

        return result

    def median_impute(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fill missing values with column median.

        Args:
            df: DataFrame to impute.
            columns: List of columns to impute. If None, imputes all numeric columns.

        Returns:
            DataFrame with missing values filled.
        """

        result: pd.DataFrame = df.copy()
        cols: list[str] = (
            columns or df.select_dtypes(include=["number"]).columns.tolist()
        )

        for col in cols:
            if col not in df.columns:
                continue

            median_val: float = df[col].median()
            if pd.isna(median_val):
                result[col] = df[col]
            else:
                result[col] = df[col].fillna(median_val)

        return result

    def mode_impute(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fill missing values with column mode.

        Args:
            df: DataFrame to impute.
            columns: List of columns to impute. If None, imputes all columns.

        Returns:
            DataFrame with missing values filled.
        """

        result: pd.DataFrame = df.copy()
        cols: list[str] = columns or df.columns.tolist()

        for col in cols:
            if col not in df.columns:
                continue

            mode_val: pd.Series = df[col].mode()
            if not mode_val.empty:
                result[col] = df[col].fillna(mode_val.iloc[0])

        return result

    def constant_impute(
        self,
        df: pd.DataFrame,
        value: float | str | None = 0,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fill missing values with a constant value.

        Args:
            df: DataFrame to impute.
            value: Constant value to fill missing values with.
            columns: List of columns to impute. If None, imputes all columns.

        Returns:
            DataFrame with missing values filled.
        """

        result: pd.DataFrame = df.copy()
        cols: list[str] = columns or df.columns.tolist()

        for col in cols:
            if col not in df.columns:
                continue

            result[col] = df[col].fillna(value)

        return result

    def knn_impute(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        n_neighbors: int = 5,
        weights: Literal["uniform", "distance"] = "uniform",
    ) -> pd.DataFrame:
        """Fill missing values using k-nearest neighbors.

        Args:
            df: DataFrame to impute.
            columns: List of columns to impute. If None, imputes all numeric columns.
            n_neighbors: Number of neighbors to use.
            weights: Weight function ('uniform' or 'distance').

        Returns:
            DataFrame with missing values filled.
        """

        result: pd.DataFrame = df.copy()
        cols: list[str] = (
            columns or df.select_dtypes(include=["number"]).columns.tolist()
        )

        if not cols:
            return result

        numeric_cols: list[str] = result.select_dtypes(
            include=["number"]
        ).columns.tolist()
        if not numeric_cols:
            return result

        data: pd.DataFrame = result[numeric_cols].copy()

        imputer: KNNImputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        imputed_data: np.ndarray = imputer.fit_transform(data)  # type: ignore

        result[numeric_cols] = imputed_data

        return result

    def model_impute(
        self,
        df: pd.DataFrame,
        target: str,
        features: list[str] | None = None,
        model: Any = None,
    ) -> pd.DataFrame:
        """Fill missing values using a regression model.

        Args:
            df: DataFrame to impute.
            target: Column name to impute.
            features: List of feature columns to use for prediction.
                      If None, uses all numeric columns except target.
            model: Scikit-learn regression model. If None, uses LinearRegression.

        Returns:
            DataFrame with missing values filled.
        """

        result: pd.DataFrame = df.copy()

        if target not in df.columns:
            return result

        if features is None:
            features = df.select_dtypes(include=["number"]).columns.tolist()
            features = [col for col in features if col != target]

        if not features:
            return result

        train_data: pd.DataFrame = df.dropna(subset=[target] + features)
        predict_data: pd.DataFrame = df[df[target].isna()]

        if train_data.empty or predict_data.empty:
            return result

        X_train: pd.DataFrame = train_data[features]
        y_train: pd.Series = train_data[target]
        x_predict: pd.DataFrame = predict_data[features]

        if model is None:
            model = LinearRegression()

        model.fit(X_train, y_train)

        predictions: np.ndarray = model.predict(x_predict)

        result.loc[result[target].isna(), target] = predictions

        return result
