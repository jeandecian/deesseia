import numpy as np
import pandas as pd
import pytest

from deesseia.utils.fake_data import FakeDataGenerator

# ============================================================
# Core fixtures
# ============================================================


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Return a sample DataFrame for testing."""

    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "age": [25, 30, 35, 40, 45],
            "salary": [50000, 60000, 70000, 80000, 90000],
            "category": ["A", "B", "A", "B", "C"],
        }
    )


@pytest.fixture
def sample_dataframe_with_missing() -> pd.DataFrame:
    """Return a sample DataFrame with missing values."""

    return pd.DataFrame(
        {
            "col1": [1, 2, None, 4, 5],
            "col2": [None, 10, 20, 30, 40],
            "col3": ["a", "b", "c", None, "e"],
            "col4": [100, 200, 300, 400, 500],
        }
    )


@pytest.fixture
def sample_dataframe_with_duplicates() -> pd.DataFrame:
    """Return a sample DataFrame with duplicates."""

    return pd.DataFrame(
        {
            "id": [1, 2, 2, 3, 3, 3],
            "value": [10, 20, 20, 30, 30, 30],
        }
    )


# ============================================================
# Scaling fixtures
# ============================================================


@pytest.fixture
def sample_dataframe_for_scaling() -> pd.DataFrame:
    """Return a sample DataFrame for scaling tests."""

    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "c": [100, 200, 300, 400, 500],
            "d": [-5, -3, 0, 3, 5],
        }
    )


@pytest.fixture
def sample_dataframe_with_outliers() -> pd.DataFrame:
    """Return a sample DataFrame with outliers for robust scaling tests."""

    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 100],
            "b": [10, 20, 30, 40, 50],
            "c": [-1000, 1, 2, 3, 4],
        }
    )


@pytest.fixture
def sample_dataframe_zero_variance() -> pd.DataFrame:
    """Return a sample DataFrame with zero variance columns."""

    return pd.DataFrame(
        {
            "constant": [1, 1, 1, 1, 1],
            "normal": [10, 20, 30, 40, 50],
            "zero_col": [0, 0, 0, 0, 0],
        }
    )


# ============================================================
# Encoding fixtures
# ============================================================


@pytest.fixture
def sample_dataframe_for_encoding() -> pd.DataFrame:
    """Return a sample DataFrame for encoding tests."""

    return pd.DataFrame(
        {
            "category": ["A", "B", "A", "C", "B", "A", "C", "C"],
            "color": ["red", "blue", "red", "green", "blue", "red", "green", "green"],
            "size": ["S", "M", "L", "S", "M", "L", "S", "M"],
            "target": [10, 20, 15, 25, 30, 12, 28, 22],
        }
    )


@pytest.fixture
def sample_dataframe_ordinal() -> pd.DataFrame:
    """Return a sample DataFrame for ordinal encoding tests."""

    return pd.DataFrame(
        {
            "education": ["high_school", "bachelor", "master", "phd", "bachelor"],
            "income": ["low", "medium", "high", "high", "medium"],
        }
    )


# ============================================================
# Imputation fixtures
# ============================================================


@pytest.fixture
def sample_dataframe_for_imputation() -> pd.DataFrame:
    """Return a sample DataFrame for imputation tests."""

    return pd.DataFrame(
        {
            "a": [1, 2, np.nan, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "c": [100, 200, 300, 400, 500],
        }
    )


@pytest.fixture
def sample_dataframe_for_model_impute() -> pd.DataFrame:
    """Return a sample DataFrame for model-based imputation tests."""

    return pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5],
            "x2": [10, 20, 30, 40, 50],
            "target": [100, 200, np.nan, 400, 500],
        }
    )


@pytest.fixture
def sample_dataframe_all_nan() -> pd.DataFrame:
    """Return a sample DataFrame with all NaN values."""

    return pd.DataFrame(
        {
            "all_nan": [np.nan, np.nan, np.nan],
            "normal": [1, 2, 3],
        }
    )


@pytest.fixture
def sample_dataframe_all_nan_model() -> pd.DataFrame:
    """Return a sample DataFrame with all NaN values for model imputation."""

    return pd.DataFrame(
        {
            "x1": [np.nan, np.nan, np.nan],
            "x2": [np.nan, np.nan, np.nan],
            "target": [np.nan, np.nan, np.nan],
        }
    )


@pytest.fixture
def sample_dataframe_only_non_numeric() -> pd.DataFrame:
    """Return a sample DataFrame with only non-numeric columns."""

    return pd.DataFrame(
        {
            "a": ["x", "y", "z"],
            "b": ["p", "q", "r"],
        }
    )


@pytest.fixture
def sample_dataframe_mixed_numeric_categorical() -> pd.DataFrame:
    """Return a sample DataFrame with mixed numeric and categorical columns."""

    return pd.DataFrame(
        {
            "a": ["x", "y", "z"],
            "b": [1, 2, np.nan],
        }
    )


@pytest.fixture
def sample_dataframe_single_target() -> pd.DataFrame:
    """Return a sample DataFrame with only a target column."""

    return pd.DataFrame(
        {
            "target": [1, 2, np.nan, 4, 5],
        }
    )


# ============================================================
# Feature creation fixtures
# ============================================================


@pytest.fixture
def sample_dataframe_poly() -> pd.DataFrame:
    """Return a sample DataFrame for polynomial features tests."""

    return pd.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "y": [2, 4, 6, 8],
        }
    )


@pytest.fixture
def sample_dataframe_poly_three() -> pd.DataFrame:
    """Return a sample DataFrame with three columns for interaction tests."""

    return pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "z": [7, 8, 9],
        }
    )


@pytest.fixture
def sample_dataframe_single_col() -> pd.DataFrame:
    """Return a sample DataFrame with a single column."""

    return pd.DataFrame({"x": [1, 2, 3]})


@pytest.fixture
def sample_dataframe_non_numeric() -> pd.DataFrame:
    """Return a sample DataFrame with non-numeric columns only."""

    return pd.DataFrame({"a": ["x", "y", "z"]})


# ============================================================
# EDA fixtures
# ============================================================


@pytest.fixture
def sample_dataframe_unique_values() -> pd.DataFrame:
    """Return a sample DataFrame with all unique values."""

    return pd.DataFrame({"a": [1, 2, 3, 4, 5]})


@pytest.fixture
def sample_dataframe_bimodal() -> pd.DataFrame:
    """Return a sample DataFrame with bimodal distribution."""

    return pd.DataFrame({"a": [1, 1, 2, 2, 3]})


@pytest.fixture
def sample_dataframe_empty_series() -> pd.DataFrame:
    """Return a sample DataFrame with an empty series (all NaN)."""

    return pd.DataFrame({"a": [np.nan, np.nan, np.nan]})


@pytest.fixture
def sample_dataframe_mixed_empty_series() -> pd.DataFrame:
    """Return a sample DataFrame with an empty series and a normal series."""

    return pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [1, 2, 3]})


@pytest.fixture
def sample_dataframe_for_skew() -> pd.DataFrame:
    """Return a sample DataFrame for skew detection tests."""

    np.random.seed(42)

    return pd.DataFrame(
        {
            "normal": np.random.normal(0, 1, 1000),
            "right_skew": np.random.exponential(1, 1000),
            "left_skew": -np.random.exponential(1, 1000),
        }
    )


@pytest.fixture
def sample_dataframe_for_outliers() -> pd.DataFrame:
    """Return a sample DataFrame for outlier detection tests."""

    np.random.seed(42)

    return pd.DataFrame(
        {
            "normal": np.random.normal(0, 1, 1000),
            "with_outliers": np.concatenate(
                [
                    np.random.normal(0, 1, 990),
                    np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
                ]
            ),
        }
    )


@pytest.fixture
def sample_dataframe_for_correlation() -> pd.DataFrame:
    """Return a sample DataFrame for correlation detection tests."""

    np.random.seed(42)

    df = pd.DataFrame(
        {
            "a": np.random.normal(0, 1, 100),
            "b": np.random.normal(0, 1, 100),
            "c": np.random.normal(0, 1, 100),
        }
    )

    df["d"] = df["a"] * 2 + np.random.normal(0, 0.1, 100)

    return df


@pytest.fixture
def sample_dataframe_many_correlations() -> pd.DataFrame:
    """Return a sample DataFrame with many high correlations."""

    np.random.seed(42)

    df = pd.DataFrame(
        {
            "a": np.random.normal(0, 1, 100),
            "b": np.random.normal(0, 1, 100),
            "c": np.random.normal(0, 1, 100),
            "d": np.random.normal(0, 1, 100),
            "e": np.random.normal(0, 1, 100),
            "f": np.random.normal(0, 1, 100),
        }
    )

    df["g"] = df["a"] * 2 + np.random.normal(0, 0.05, 100)
    df["h"] = df["b"] * 2 + np.random.normal(0, 0.05, 100)
    df["i"] = df["c"] * 2 + np.random.normal(0, 0.05, 100)
    df["j"] = df["d"] * 2 + np.random.normal(0, 0.05, 100)

    return df


@pytest.fixture
def sample_dataframe_clean() -> pd.DataFrame:
    """Return a clean sample DataFrame with no issues (no skew, no outliers, no correlations)."""

    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 1, 5, 3],
        }
    )


@pytest.fixture
def sample_dataframe_high_correlation() -> pd.DataFrame:
    """Return a sample DataFrame with high correlation between columns."""

    np.random.seed(42)

    df = pd.DataFrame(
        {
            "a": np.random.normal(0, 1, 100),
            "b": np.random.normal(0, 1, 100),
        }
    )

    df["c"] = df["a"] * 2 + np.random.normal(0, 0.1, 100)

    return df


@pytest.fixture
def sample_dataframe_single_column() -> pd.DataFrame:
    """Return a sample DataFrame with a single numeric column."""

    return pd.DataFrame({"a": [1, 2, 3, 4, 5]})


# ============================================================
# Stats fixtures
# ============================================================


@pytest.fixture
def sample_data_ttest() -> np.ndarray:
    """Return sample data for one-sample t-test and paired t-test (before)."""

    data: np.ndarray = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    return data


@pytest.fixture
def sample_data_paired_after() -> np.ndarray:
    """Return sample data for paired t-test (after)."""

    np.random.seed(42)

    data: np.ndarray = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + np.random.normal(
        0, 0.5, 10
    )

    return data


@pytest.fixture
def sample_data_group1() -> np.ndarray:
    """Return sample data for group 1 (independent t-test, ANOVA, Mann-Whitney)."""

    np.random.seed(42)

    data: np.ndarray = np.random.normal(0, 1, 30)

    return data


@pytest.fixture
def sample_data_group2() -> np.ndarray:
    """Return sample data for group 2 (independent t-test, ANOVA, Mann-Whitney)."""

    np.random.seed(42)

    data: np.ndarray = np.random.normal(0.5, 1, 30)

    return data


@pytest.fixture
def sample_data_group3() -> np.ndarray:
    """Return sample data for group 3 (ANOVA)."""

    np.random.seed(42)

    data: np.ndarray = np.random.normal(1, 1, 30)

    return data


@pytest.fixture
def sample_data_group2_welch() -> np.ndarray:
    """Return sample data for group 2 (Welch's t-test with different variance)."""

    np.random.seed(42)

    data: np.ndarray = np.random.normal(0.5, 2, 30)

    return data


@pytest.fixture
def sample_data_chi_square() -> np.ndarray:
    """Return observed contingency table for chi-square test."""

    data: np.ndarray = np.array([[10, 20], [30, 40]])

    return data


@pytest.fixture
def sample_data_single_observation() -> np.ndarray:
    """Return a single observation for insufficient data tests."""

    data: np.ndarray = np.array([1])

    return data


@pytest.fixture
def sample_data_two_observations() -> np.ndarray:
    """Return two observations for insufficient data tests."""

    data: np.ndarray = np.array([2, 3])

    return data


@pytest.fixture
def sample_data_three_observations() -> np.ndarray:
    """Return three observations for mismatched length tests."""

    data: np.ndarray = np.array([1, 2, 3])

    return data


@pytest.fixture
def sample_data_two_observations_mismatch() -> np.ndarray:
    """Return two observations for mismatched length tests."""

    data: np.ndarray = np.array([4, 5])

    return data


# ============================================================
# P-value interpretation test fixtures
# ============================================================


@pytest.fixture
def sample_data_very_strong_evidence() -> np.ndarray:
    """Return sample data with p < 0.001."""

    np.random.seed(42)

    data: np.ndarray = np.random.normal(1.5, 1.0, 100)

    return data


@pytest.fixture
def sample_data_moderate_evidence() -> np.ndarray:
    """Return sample data with 0.001 <= p < 0.01."""

    np.random.seed(123)

    data: np.ndarray = np.random.normal(0.8, 1.0, 20)

    return data


@pytest.fixture
def sample_data_strong_evidence() -> np.ndarray:
    """Return sample data with 0.01 <= p < 0.05."""

    np.random.seed(42)

    data: np.ndarray = np.random.normal(0.6, 1.0, 30)

    return data


@pytest.fixture
def sample_data_no_evidence() -> np.ndarray:
    """Return sample data with p >= 0.05."""

    np.random.seed(42)

    data: np.ndarray = np.random.normal(0.0, 1.0, 20)

    return data


@pytest.fixture
def sample_group1_very_strong_evidence() -> np.ndarray:
    """Return group 1 for independent t-test with p < 0.001."""

    np.random.seed(42)

    data: np.ndarray = np.random.normal(0, 1, 50)

    return data


@pytest.fixture
def sample_group2_very_strong_evidence() -> np.ndarray:
    """Return group 2 for independent t-test with p < 0.001."""

    np.random.seed(42)

    data: np.ndarray = np.random.normal(1.8, 1, 50)

    return data


@pytest.fixture
def sample_group1_moderate_evidence() -> np.ndarray:
    """Return group 1 for independent t-test with 0.001 <= p < 0.01."""

    np.random.seed(999)

    data: np.ndarray = np.random.normal(0, 1, 20)

    return data


@pytest.fixture
def sample_group2_moderate_evidence() -> np.ndarray:
    """Return group 2 for independent t-test with 0.001 <= p < 0.01."""

    np.random.seed(999)

    data: np.ndarray = np.random.normal(1.0, 1, 20)

    return data


@pytest.fixture
def sample_chi_square_strong_evidence() -> np.ndarray:
    """Return contingency table with p < 0.001 for chi-square test."""

    data: np.ndarray = np.array([[50, 5], [5, 50]])

    return data


@pytest.fixture
def sample_group1_mann_whitney_no_evidence() -> np.ndarray:
    """Return group 1 for Mann-Whitney test with p >= 0.05."""

    np.random.seed(42)

    data: np.ndarray = np.random.normal(0, 1, 20)

    return data


@pytest.fixture
def sample_group2_mann_whitney_no_evidence() -> np.ndarray:
    """Return group 2 for Mann-Whitney test with p >= 0.05."""

    np.random.seed(42)

    data: np.ndarray = np.random.normal(0.3, 1, 20)

    return data


# ============================================================
# Split fixtures
# ============================================================


@pytest.fixture
def sample_dataframe_for_split() -> pd.DataFrame:
    """Return a sample DataFrame for split tests."""

    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )


@pytest.fixture
def sample_target_for_split() -> pd.Series:
    """Return a sample target Series for split tests."""

    return pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


@pytest.fixture
def sample_target_binary_for_split() -> pd.Series:
    """Return a sample binary target Series for stratified split tests."""

    return pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])


# ============================================================
# Utility fixtures
# ============================================================


@pytest.fixture
def fake_data_generator() -> FakeDataGenerator:
    """Return a seeded FakeDataGenerator for reproducible tests."""

    return FakeDataGenerator(random_state=42)
