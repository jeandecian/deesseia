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
def fake_data_generator() -> FakeDataGenerator:
    """Return a seeded FakeDataGenerator for reproducible tests."""

    return FakeDataGenerator(random_state=42)
