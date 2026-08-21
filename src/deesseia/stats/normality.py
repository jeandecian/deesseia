from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore[import-untyped]


class Normality:
    """Normality tests for statistical analysis."""

    def __init__(self) -> None:
        """Initialize the Normality class."""

    def shapiro_wilk(
        self,
        data: pd.Series | np.ndarray,
    ) -> dict[str, Any]:
        """Perform Shapiro-Wilk test for normality.

        Args:
            data: Sample data to test for normality.

        Returns:
            Dictionary containing test statistic, p-value, and conclusion.
        """

        if len(data) < 3:
            raise ValueError(
                "Sample must have at least 3 observations for Shapiro-Wilk test."
            )

        result = stats.shapiro(data)  # type: ignore
        statistic, p_value = cast(tuple[float, float], result)

        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "conclusion": (
                "Data is not normally distributed (reject H0)"
                if p_value < 0.05
                else "Data appears normally distributed (fail to reject H0)"
            ),
        }

    def kolmogorov_smirnov(
        self,
        data: pd.Series | np.ndarray,
        distribution: Literal["norm", "uniform", "exponential"] = "norm",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Perform Kolmogorov-Smirnov test for goodness of fit.

        Args:
            data: Sample data to test.
            distribution: Distribution to test against ('norm', 'uniform', 'exponential').
            **kwargs: Additional parameters for the distribution (e.g., loc, scale for norm).

        Returns:
            Dictionary containing test statistic, p-value, and conclusion.
        """

        if len(data) < 2:
            raise ValueError(
                "Sample must have at least 2 observations for Kolmogorov-Smirnov test."
            )

        dist_map: dict[str, Any] = {
            "norm": stats.norm,
            "uniform": stats.uniform,
            "exponential": stats.expon,
        }

        if distribution not in dist_map:
            raise ValueError(
                f"Distribution must be one of {list(dist_map.keys())}, got '{distribution}'"
            )

        dist: Any = dist_map[distribution]

        params: Any = dist.fit(data)

        result = stats.kstest(data, dist.cdf, args=params)  # type: ignore
        statistic, p_value = cast(tuple[float, float], result)

        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "distribution": distribution,
            "significant": p_value < 0.05,
            "conclusion": (
                f"Data does not follow {distribution} distribution (reject H0)"
                if p_value < 0.05
                else f"Data appears to follow {distribution} distribution (fail to reject H0)"
            ),
        }

    def d_agostino_pearson(
        self,
        data: pd.Series | np.ndarray,
    ) -> dict[str, Any]:
        """Perform D'Agostino-Pearson test for normality (based on skewness and kurtosis).

        Args:
            data: Sample data to test for normality.

        Returns:
            Dictionary containing test statistic, p-value, and conclusion.
        """

        if len(data) < 8:
            raise ValueError(
                "Sample must have at least 8 observations for D'Agostino-Pearson test."
            )

        result = stats.normaltest(data)  # type: ignore
        statistic, p_value = cast(tuple[float, float], result)

        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "conclusion": (
                "Data is not normally distributed (reject H0)"
                if p_value < 0.05
                else "Data appears normally distributed (fail to reject H0)"
            ),
        }
