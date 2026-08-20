from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd


class FakeDataGenerator:
    """Generate synthetic/fake datasets for testing."""

    def __init__(self, random_state: int | None = None) -> None:
        """Initialize the generator with an optional random seed.

        Args:
            random_state: Seed for reproducible random generation.
        """

        self.random_state: int | None = random_state
        self.rng: np.random.Generator = np.random.default_rng(random_state)

        if random_state is not None:
            random.seed(random_state)

    def generate_numeric(
        self,
        n_samples: int = 100,
        n_features: int = 5,
        distribution: str = "normal",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate numeric dataset.

        Args:
            n_samples: Number of samples/rows.
            n_features: Number of features/columns.
            distribution: Distribution type ('normal', 'uniform', 'exponential', 'poisson').
            **kwargs: Distribution-specific parameters.

        Returns:
            DataFrame with generated numeric data.
        """

        data: dict[str, np.ndarray] = {}

        for i in range(n_features):
            if distribution == "normal":
                data[f"feature_{i+1}"] = self.rng.normal(
                    loc=kwargs.get("loc", 0),
                    scale=kwargs.get("scale", 1),
                    size=n_samples,
                )
            elif distribution == "uniform":
                data[f"feature_{i+1}"] = self.rng.uniform(
                    low=kwargs.get("low", 0),
                    high=kwargs.get("high", 1),
                    size=n_samples,
                )
            elif distribution == "exponential":
                data[f"feature_{i+1}"] = self.rng.exponential(
                    scale=kwargs.get("scale", 1),
                    size=n_samples,
                )
            elif distribution == "poisson":
                data[f"feature_{i+1}"] = self.rng.poisson(
                    lam=kwargs.get("lam", 1),
                    size=n_samples,
                )
            else:
                raise ValueError(f"Unsupported distribution: {distribution}")

        df: pd.DataFrame = pd.DataFrame(data)

        return df

    def generate_categorical(
        self,
        n_samples: int = 100,
        n_features: int = 3,
        categories: list[str] | None = None,
    ) -> pd.DataFrame:
        """Generate categorical dataset.

        Args:
            n_samples: Number of samples/rows.
            n_features: Number of features/columns.
            categories: List of category values to choose from.

        Returns:
            DataFrame with generated categorical data.
        """

        if categories is None:
            categories = ["A", "B", "C", "D"]

        data: dict[str, list[str]] = {}

        for i in range(n_features):
            data[f"category_{i+1}"] = [
                self.rng.choice(categories) for _ in range(n_samples)
            ]

        df: pd.DataFrame = pd.DataFrame(data)

        return df

    def generate_mixed(
        self,
        n_samples: int = 100,
        n_numeric: int = 3,
        n_categorical: int = 2,
    ) -> pd.DataFrame:
        """Generate mixed numeric and categorical dataset.

        Args:
            n_samples: Number of samples/rows.
            n_numeric: Number of numeric features.
            n_categorical: Number of categorical features.

        Returns:
            DataFrame with mixed data types.
        """

        numeric_df: pd.DataFrame = self.generate_numeric(n_samples, n_numeric)
        categorical_df: pd.DataFrame = self.generate_categorical(
            n_samples, n_categorical
        )

        return pd.concat([numeric_df, categorical_df], axis=1)

    def generate_missing_data(
        self,
        n_rows: int = 100,
        n_cols: int = 5,
        missing_probability: float = 0.1,
    ) -> pd.DataFrame:
        """Generate data with missing values.

        Args:
            n_rows: Number of rows.
            n_cols: Number of columns.
            missing_probability: Probability of a cell being missing (0.0 to 1.0).

        Returns:
            DataFrame with missing values introduced.
        """

        df: pd.DataFrame = self.generate_numeric(n_samples=n_rows, n_features=n_cols)
        mask: np.ndarray = self.rng.random(df.shape) < missing_probability

        masked_df: pd.DataFrame = pd.DataFrame(df.mask(mask))

        return masked_df
