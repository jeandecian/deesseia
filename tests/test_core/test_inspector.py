from typing import Any

import pandas as pd
import pytest  # noqa: F401 # type: ignore[import-unused]

from deesseia.core.inspector import DataInspector


class TestDataInspector:
    """Test data inspection functionality."""

    def test_summary(self, sample_dataframe: pd.DataFrame):
        """Test summary generation."""

        inspector: DataInspector = DataInspector()
        summary: dict[str, Any] = inspector.summary(sample_dataframe)

        assert "shape" in summary
        assert "columns" in summary
        assert "dtypes" in summary
        assert "missing" in summary
        assert "missing_percentage" in summary
        assert "memory_usage" in summary
        assert "total_memory" in summary
        assert "describe" in summary
        assert summary["shape"] == (5, 5)

    def test_head(self, sample_dataframe: pd.DataFrame):
        """Test head method."""

        inspector: DataInspector = DataInspector()
        result: pd.DataFrame = inspector.head(sample_dataframe, n=2)

        assert len(result) == 2
        assert isinstance(result, pd.DataFrame)

    def test_tail(self, sample_dataframe: pd.DataFrame):
        """Test tail method."""

        inspector: DataInspector = DataInspector()
        result: pd.DataFrame = inspector.tail(sample_dataframe, n=2)

        assert len(result) == 2
        assert isinstance(result, pd.DataFrame)

    def test_shape(self, sample_dataframe: pd.DataFrame):
        """Test shape method."""

        inspector: DataInspector = DataInspector()
        result: tuple[int, int] = inspector.shape(sample_dataframe)

        assert result == (5, 5)

    def test_columns(self, sample_dataframe: pd.DataFrame):
        """Test columns method."""

        inspector: DataInspector = DataInspector()
        result: list[str] = inspector.columns(sample_dataframe)

        assert result == ["id", "name", "age", "salary", "category"]

    def test_dtypes(self, sample_dataframe: pd.DataFrame):
        """Test dtypes method."""

        inspector: DataInspector = DataInspector()
        result: dict[str, str] = inspector.dtypes(sample_dataframe)

        assert "id" in result
        assert "name" in result
        assert "age" in result
        assert isinstance(result, dict)

    def test_memory_usage(self, sample_dataframe: pd.DataFrame):
        """Test memory_usage method."""

        inspector: DataInspector = DataInspector()
        result: dict[str, int] = inspector.memory_usage(sample_dataframe)

        assert isinstance(result, dict)
        assert "id" in result
