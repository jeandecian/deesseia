from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from deesseia.core.loader import DataLoader


class TestDataLoader:
    """Test data loading functionality."""

    def test_from_csv(self, tmp_path: Path):
        """Test loading CSV file."""

        csv_path: Path = tmp_path / "test.csv"
        pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]}).to_csv(csv_path, index=False)

        df: pd.DataFrame = DataLoader.from_csv(str(csv_path))

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)
        assert list(df.columns) == ["col1", "col2"]

    def test_from_csv_with_delimiter(self, tmp_path: Path):
        """Test loading CSV with custom delimiter."""

        csv_path: Path = tmp_path / "test.tsv"
        pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]}).to_csv(
            csv_path, index=False, sep="\t"
        )

        df: pd.DataFrame = DataLoader.from_csv(str(csv_path), delimiter="\t")

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)

    def test_from_csv_with_kwargs(self, tmp_path: Path):
        """Test loading CSV with additional pandas kwargs."""

        csv_path: Path = tmp_path / "test.csv"
        pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]}).to_csv(csv_path, index=False)

        df: pd.DataFrame = DataLoader.from_csv(str(csv_path), dtype={"col1": "float64"})

        assert df["col1"].dtype == "float64"

    def test_from_json(self, tmp_path: Path):
        """Test loading JSON file."""

        json_path: Path = tmp_path / "test.json"
        pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]}).to_json(json_path)

        df: pd.DataFrame = DataLoader.from_json(str(json_path))

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)

    def test_from_json_with_orient(self, tmp_path: Path):
        """Test loading JSON file with custom orientation."""

        json_path: Path = tmp_path / "test.json"
        data: list[dict[str, Any]] = [
            {"col1": 1, "col2": "a"},
            {"col1": 2, "col2": "b"},
        ]
        pd.DataFrame(data).to_json(json_path, orient="records")

        df: pd.DataFrame = DataLoader.from_json(str(json_path), orient="records")

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)

    def test_from_parquet(self, tmp_path: Path):
        """Test loading Parquet file."""

        parquet_path: Path = tmp_path / "test.parquet"
        pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]}).to_parquet(parquet_path)

        df: pd.DataFrame = DataLoader.from_parquet(str(parquet_path))

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)

    def test_from_sql(self, tmp_path: Path):
        """Test loading from SQL database."""

        import sqlite3

        db_path: Path = tmp_path / "test.db"
        conn: sqlite3.Connection = sqlite3.connect(str(db_path))
        pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]}).to_sql(
            "test_table", conn, if_exists="replace", index=False
        )
        conn.close()

        df: pd.DataFrame = DataLoader.from_sql(
            "SELECT * FROM test_table",
            f"sqlite:///{db_path}",
        )

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)

    def test_from_sql_with_where_clause(self, tmp_path: Path):
        """Test loading from SQL with WHERE clause."""

        import sqlite3

        db_path: Path = tmp_path / "test.db"
        conn: sqlite3.Connection = sqlite3.connect(str(db_path))
        pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]}).to_sql(
            "test_table", conn, if_exists="replace", index=False
        )
        conn.close()

        df: pd.DataFrame = DataLoader.from_sql(
            "SELECT * FROM test_table WHERE col1 > 1",
            f"sqlite:///{db_path}",
        )

        assert len(df) == 2

    def test_load_not_implemented(self):
        """Test that the base load method raises NotImplementedError."""

        loader: DataLoader = DataLoader()
        with pytest.raises(NotImplementedError):
            loader.load()
