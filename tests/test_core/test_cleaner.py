import pandas as pd
import pytest

from deesseia.core.cleaner import Cleaner


class TestCleaner:
    """Test data cleaning functionality."""

    def test_init(self, sample_dataframe: pd.DataFrame):
        """Test cleaner initialization."""

        cleaner: Cleaner = Cleaner(sample_dataframe)

        assert isinstance(cleaner.df, pd.DataFrame)
        assert not cleaner.df is sample_dataframe  # Should be a copy

    def test_handle_missing_mean(self, sample_dataframe_with_missing: pd.DataFrame):
        """Test missing value imputation with mean."""

        cleaner: Cleaner = Cleaner(sample_dataframe_with_missing)
        result: pd.DataFrame = cleaner.handle_missing(strategy="mean")

        # col1 had one None, should be filled with mean of [1,2,4,5] = 3.0
        assert result["col1"].iloc[2] == 3.0

    def test_handle_missing_median(self, sample_dataframe_with_missing: pd.DataFrame):
        """Test missing value imputation with median."""

        cleaner: Cleaner = Cleaner(sample_dataframe_with_missing)
        result: pd.DataFrame = cleaner.handle_missing(strategy="median")
        numeric_cols: pd.Index = result.select_dtypes(include=["number"]).columns

        assert not result[numeric_cols].isnull().any().any()

    def test_handle_missing_ffill(self, sample_dataframe_with_missing: pd.DataFrame):
        """Test missing value imputation with forward fill."""

        cleaner: Cleaner = Cleaner(sample_dataframe_with_missing)
        result: pd.DataFrame = cleaner.handle_missing(strategy="ffill")

        # col1: [1, 2, 2, 4, 5]
        assert result["col1"].iloc[2] == 2.0

    def test_handle_missing_bfill(self, sample_dataframe_with_missing: pd.DataFrame):
        """Test missing value imputation with backward fill."""

        cleaner: Cleaner = Cleaner(sample_dataframe_with_missing)
        result: pd.DataFrame = cleaner.handle_missing(strategy="bfill")

        # col1: [1, 2, 4, 4, 5]
        assert result["col1"].iloc[2] == 4.0

    def test_handle_missing_drop(self, sample_dataframe_with_missing: pd.DataFrame):
        """Test dropping rows with missing values."""

        cleaner: Cleaner = Cleaner(sample_dataframe_with_missing)
        result: pd.DataFrame = cleaner.handle_missing(strategy="drop")
        numeric_cols: pd.Index = result.select_dtypes(include=["number"]).columns

        assert len(result) == 3  # Kept rows: 1, 3, 4 = 3 rows
        assert not result[numeric_cols].isnull().any().any()

    def test_drop_duplicates(self, sample_dataframe_with_duplicates: pd.DataFrame):
        """Test duplicate removal."""

        cleaner: Cleaner = Cleaner(sample_dataframe_with_duplicates)
        result: pd.DataFrame = cleaner.drop_duplicates()

        assert len(result) == 3  # 1, 2, 3 (unique ids)

    def test_drop_duplicates_with_subset(
        self, sample_dataframe_with_duplicates: pd.DataFrame
    ):
        """Test duplicate removal with subset."""

        cleaner: Cleaner = Cleaner(sample_dataframe_with_duplicates)
        result: pd.DataFrame = cleaner.drop_duplicates(subset=["id"])

        assert len(result) == 3

    def test_drop_columns(self, sample_dataframe: pd.DataFrame):
        """Test dropping columns."""

        cleaner: Cleaner = Cleaner(sample_dataframe)
        result: pd.DataFrame = cleaner.drop_columns(["id", "name"])

        assert "id" not in result.columns
        assert "name" not in result.columns
        assert "age" in result.columns

    def test_drop_empty_columns(self):
        """Test dropping empty columns."""

        df: pd.DataFrame = pd.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": [None, None, None],
                "col3": ["a", "b", "c"],
            }
        )
        cleaner: Cleaner = Cleaner(df)
        result: pd.DataFrame = cleaner.drop_empty_columns()

        assert "col2" not in result.columns
        assert "col1" in result.columns
        assert "col3" in result.columns

    def test_drop_single_cardinality_columns(self):
        """Test dropping single-cardinality columns."""

        df: pd.DataFrame = pd.DataFrame(
            {
                "col1": [1, 1, 1],
                "col2": ["a", "b", "c"],
                "col3": [True, True, True],
            }
        )
        cleaner: Cleaner = Cleaner(df)
        result: pd.DataFrame = cleaner.drop_single_cardinality_columns()

        assert "col1" not in result.columns
        assert "col3" not in result.columns
        assert "col2" in result.columns

    def test_standardize_column_names(self):
        """Test column name standardization."""

        df: pd.DataFrame = pd.DataFrame(
            {
                "First Name": [1, 2],
                "Last-Name": [3, 4],
                "AGE_GROUP": [5, 6],
            }
        )
        cleaner: Cleaner = Cleaner(df)
        result: pd.DataFrame = cleaner.standardize_column_names()

        assert list(result.columns) == ["first_name", "last_name", "age_group"]

    def test_rename_columns(self, sample_dataframe: pd.DataFrame):
        """Test renaming columns."""

        cleaner: Cleaner = Cleaner(sample_dataframe)
        result: pd.DataFrame = cleaner.rename_columns(
            {"id": "identifier", "name": "full_name"}
        )

        assert "identifier" in result.columns
        assert "full_name" in result.columns
        assert "id" not in result.columns
        assert "name" not in result.columns

    def test_select_columns(self, sample_dataframe: pd.DataFrame):
        """Test selecting columns."""

        cleaner: Cleaner = Cleaner(sample_dataframe)
        result: pd.DataFrame = cleaner.select_columns(["id", "name"])

        assert list(result.columns) == ["id", "name"]
        assert isinstance(result, pd.DataFrame)

    def test_handle_missing_unsupported_strategy(
        self, sample_dataframe_with_missing: pd.DataFrame
    ):
        """Test unsupported strategy raises error."""

        cleaner: Cleaner = Cleaner(sample_dataframe_with_missing)
        with pytest.raises(ValueError, match="Unsupported strategy: invalid"):
            cleaner.handle_missing(strategy="invalid")

    def test_handle_missing_with_columns(
        self, sample_dataframe_with_missing: pd.DataFrame
    ):
        """Test handling missing values on specific columns."""

        cleaner: Cleaner = Cleaner(sample_dataframe_with_missing)
        result: pd.DataFrame = cleaner.handle_missing(
            strategy="mean", columns=["col1", "col2"]
        )

        assert not result["col1"].isnull().any()
        assert not result["col2"].isnull().any()
        # col3 should still have missing values
        assert result["col3"].isnull().any()
