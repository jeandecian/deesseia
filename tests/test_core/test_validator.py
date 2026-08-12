import pandas as pd
import pytest  # noqa: F401 # type: ignore[import-unused]

from deesseia.core.validator import Validator


class TestValidator:
    """Test data validation functionality."""

    def test_validate_schema_valid(self, sample_dataframe: pd.DataFrame):
        """Test schema validation with valid data."""

        validator: Validator = Validator()
        schema: dict[str, str] = {
            "id": str(sample_dataframe["id"].dtype),
            "name": str(sample_dataframe["name"].dtype),
            "age": str(sample_dataframe["age"].dtype),
            "salary": str(sample_dataframe["salary"].dtype),
            "category": str(sample_dataframe["category"].dtype),
        }
        errors: dict[str, list[str]] = validator.validate_schema(
            sample_dataframe, schema
        )

        assert errors["missing_columns"] == []
        assert errors["wrong_dtypes"] == []

    def test_validate_schema_missing_columns(self, sample_dataframe: pd.DataFrame):
        """Test schema validation with missing columns."""

        validator: Validator = Validator()
        schema: dict[str, str] = {
            "id": "int64",
            "name": "object",
            "age": "int64",
            "salary": "int64",
            "category": "object",
            "missing_col": "int64",
        }
        errors: dict[str, list[str]] = validator.validate_schema(
            sample_dataframe, schema
        )

        assert "missing_col" in errors["missing_columns"]

    def test_validate_schema_wrong_dtypes(self, sample_dataframe: pd.DataFrame):
        """Test schema validation with wrong data types."""

        validator: Validator = Validator()
        schema: dict[str, str] = {
            "id": "float64",  # Should be int64
            "name": "object",
            "age": "int64",
            "salary": "int64",
            "category": "object",
        }
        errors: dict[str, list[str]] = validator.validate_schema(
            sample_dataframe, schema
        )

        assert "id" in errors["wrong_dtypes"]

    def test_check_missing_values(self, sample_dataframe_with_missing: pd.DataFrame):
        """Test missing values check."""

        validator: Validator = Validator()
        result: dict[str, float] = validator.check_missing_values(
            sample_dataframe_with_missing
        )

        assert "col1" in result
        assert "col2" in result
        assert "col3" in result
        assert "col4" in result
        assert result["col1"] == 20.0  # 1 missing out of 5 = 20%
        assert result["col2"] == 20.0
        assert result["col3"] == 20.0
        assert result["col4"] == 0.0

    def test_check_missing_values_no_missing(self, sample_dataframe: pd.DataFrame):
        """Test missing values check with no missing data."""

        validator: Validator = Validator()
        result: dict[str, float] = validator.check_missing_values(sample_dataframe)

        assert all(v == 0.0 for v in result.values())

    def test_check_duplicates(self, sample_dataframe_with_duplicates: pd.DataFrame):
        """Test duplicate check."""

        validator: Validator = Validator()
        result: dict[str, int] = validator.check_duplicates(
            sample_dataframe_with_duplicates
        )

        assert "total_duplicates" in result
        assert result["total_duplicates"] == 3

    def test_check_duplicates_no_duplicates(self, sample_dataframe: pd.DataFrame):
        """Test duplicate check with no duplicates."""

        validator: Validator = Validator()
        result: dict[str, int] = validator.check_duplicates(sample_dataframe)

        assert result["total_duplicates"] == 0
