import pandas as pd
import pytest

from deesseia.utils.fake_data import FakeDataGenerator


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
def sample_dataframe_single_value_column() -> pd.DataFrame:
    """Return a sample DataFrame with a column of identical values."""

    return pd.DataFrame(
        {
            "a": [1, 1, 1, 1, 1],
            "b": [1, 2, 3, 4, 5],
        }
    )


@pytest.fixture
def sample_dataframe_all_zeros() -> pd.DataFrame:
    """Return a sample DataFrame with all zeros in one column."""

    return pd.DataFrame(
        {
            "a": [0, 0, 0, 0, 0],
            "b": [1, 2, 3, 4, 5],
        }
    )


@pytest.fixture
def sample_dataframe_for_log_transform() -> pd.DataFrame:
    """Return a sample DataFrame for log transform tests."""

    return pd.DataFrame(
        {
            "a": [1, 10, 100, 1000],
            "b": [2, 4, 8, 16],
        }
    )


@pytest.fixture
def sample_dataframe_with_zeros() -> pd.DataFrame:
    """Return a sample DataFrame with zeros for log transform tests."""

    return pd.DataFrame({"a": [0, 1, 2, 3]})


@pytest.fixture
def sample_dataframe_zero_iqr() -> pd.DataFrame:
    """Return a sample DataFrame with a column where IQR = 0."""

    return pd.DataFrame(
        {
            "a": [1, 1, 1, 1, 1],  # All same, IQR = 0
            "b": [10, 20, 30, 40, 50],
        }
    )


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


@pytest.fixture
def fake_data_generator() -> FakeDataGenerator:
    """Return a seeded FakeDataGenerator for reproducible tests."""

    return FakeDataGenerator(random_state=42)
