from typing import Any

import numpy as np
import pytest

from deesseia.stats.normality import Normality


class TestNormality:
    """Test normality testing functionality."""

    def test_shapiro_wilk_normal_data(self, sample_data_normal: np.ndarray) -> None:
        """Test Shapiro-Wilk test with normally distributed data."""

        norm: Normality = Normality()
        result: dict[str, Any] = norm.shapiro_wilk(sample_data_normal)

        assert "statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert "conclusion" in result
        assert result["p_value"] > 0.05
        assert "normally distributed" in result["conclusion"]

    def test_shapiro_wilk_non_normal_data(
        self, sample_data_non_normal: np.ndarray
    ) -> None:
        """Test Shapiro-Wilk test with non-normally distributed data."""

        norm: Normality = Normality()
        result: dict[str, Any] = norm.shapiro_wilk(sample_data_non_normal)

        assert "statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert "conclusion" in result
        assert result["p_value"] < 0.05
        assert "not normally distributed" in result["conclusion"]

    def test_shapiro_wilk_insufficient_data(
        self, sample_data_shapiro_insufficient: np.ndarray
    ) -> None:
        """Test Shapiro-Wilk test with insufficient data."""

        norm: Normality = Normality()

        with pytest.raises(
            ValueError, match="Sample must have at least 3 observations"
        ):
            norm.shapiro_wilk(sample_data_shapiro_insufficient)

    def test_kolmogorov_smirnov_normal(self, sample_data_normal: np.ndarray) -> None:
        """Test Kolmogorov-Smirnov test with normal distribution."""

        norm: Normality = Normality()
        result: dict[str, Any] = norm.kolmogorov_smirnov(
            sample_data_normal, distribution="norm"
        )

        assert "statistic" in result
        assert "p_value" in result
        assert "distribution" in result
        assert "significant" in result
        assert "conclusion" in result
        assert result["p_value"] > 0.05
        assert "follow norm distribution" in result["conclusion"]

    def test_kolmogorov_smirnov_uniform(self, sample_data_uniform: np.ndarray) -> None:
        """Test Kolmogorov-Smirnov test with uniform distribution."""

        norm: Normality = Normality()
        result: dict[str, Any] = norm.kolmogorov_smirnov(
            sample_data_uniform, distribution="uniform"
        )

        assert "statistic" in result
        assert "p_value" in result
        assert result["p_value"] > 0.05
        assert "follow uniform distribution" in result["conclusion"]

    def test_kolmogorov_smirnov_exponential(
        self, sample_data_exponential: np.ndarray
    ) -> None:
        """Test Kolmogorov-Smirnov test with exponential distribution."""

        norm: Normality = Normality()
        result: dict[str, Any] = norm.kolmogorov_smirnov(
            sample_data_exponential, distribution="exponential"
        )

        assert "statistic" in result
        assert "p_value" in result
        assert result["p_value"] > 0.05
        assert "follow exponential distribution" in result["conclusion"]

    def test_kolmogorov_smirnov_invalid_distribution(
        self, sample_data_normal: np.ndarray
    ) -> None:
        """Test Kolmogorov-Smirnov test with invalid distribution."""

        norm: Normality = Normality()

        with pytest.raises(ValueError, match="Distribution must be one of"):
            norm.kolmogorov_smirnov(sample_data_normal, distribution="invalid")  # type: ignore

    def test_kolmogorov_smirnov_insufficient_data(
        self, sample_data_ks_insufficient: np.ndarray
    ) -> None:
        """Test Kolmogorov-Smirnov test with insufficient data."""

        norm: Normality = Normality()

        with pytest.raises(
            ValueError, match="Sample must have at least 2 observations"
        ):
            norm.kolmogorov_smirnov(sample_data_ks_insufficient)

    def test_d_agostino_pearson_normal(self, sample_data_normal: np.ndarray) -> None:
        """Test D'Agostino-Pearson test with normally distributed data."""

        norm: Normality = Normality()
        result: dict[str, Any] = norm.d_agostino_pearson(sample_data_normal)

        assert "statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert "conclusion" in result
        assert result["p_value"] > 0.05
        assert "normally distributed" in result["conclusion"]

    def test_d_agostino_pearson_non_normal(
        self, sample_data_non_normal: np.ndarray
    ) -> None:
        """Test D'Agostino-Pearson test with non-normally distributed data."""

        norm: Normality = Normality()
        result: dict[str, Any] = norm.d_agostino_pearson(sample_data_non_normal)

        assert "statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert "conclusion" in result
        assert result["p_value"] < 0.05
        assert "not normally distributed" in result["conclusion"]

    def test_d_agostino_pearson_insufficient_data(
        self, sample_data_dagostino_insufficient: np.ndarray
    ) -> None:
        """Test D'Agostino-Pearson test with insufficient data."""

        norm: Normality = Normality()

        with pytest.raises(
            ValueError, match="Sample must have at least 8 observations"
        ):
            norm.d_agostino_pearson(sample_data_dagostino_insufficient)
