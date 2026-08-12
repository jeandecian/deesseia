from __future__ import annotations

from typing import Any, Literal, cast

import pandas as pd

from deesseia.core.base.loader import BaseDataLoader


class DataLoader(BaseDataLoader):
    """Load data from various file formats."""

    def load(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Load data from a source.

        This is a convenience method that delegates to the appropriate
        from_* method based on the file extension or source type.

        Raises:
            NotImplementedError: Until file type detection is implemented.
        """

        raise NotImplementedError("Use specific from_* methods instead.")

    @staticmethod
    def from_csv(
        filepath: str,
        delimiter: str = ",",
        encoding: str = "utf-8",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load data from CSV file.

        Args:
            filepath: Path to the CSV file.
            delimiter: Delimiter used in the CSV file.
            encoding: File encoding.
            **kwargs: Additional arguments passed to pandas.read_csv.

        Returns:
            DataFrame containing the loaded data.
        """

        return cast(
            pd.DataFrame,
            pd.read_csv(filepath, sep=delimiter, encoding=encoding, **kwargs),
        )

    @staticmethod
    def from_json(
        filepath: str,
        orient: Literal[
            "split", "records", "index", "columns", "values", "table"
        ] = "records",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load data from JSON file.

        Args:
            filepath: Path to the JSON file.
            orient: JSON orientation format.
            **kwargs: Additional arguments passed to pandas.read_json.

        Returns:
            DataFrame containing the loaded data.
        """

        return cast(pd.DataFrame, pd.read_json(filepath, orient=orient, **kwargs))

    @staticmethod
    def from_parquet(
        filepath: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load data from Parquet file.

        Args:
            filepath: Path to the Parquet file.
            **kwargs: Additional arguments passed to pandas.read_parquet.

        Returns:
            DataFrame containing the loaded data.
        """

        return pd.read_parquet(filepath, **kwargs)

    @staticmethod
    def from_sql(
        query: str,
        connection: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load data from SQL query.

        Args:
            query: SQL query to execute.
            connection: SQLAlchemy connection string.
            **kwargs: Additional arguments passed to pandas.read_sql.

        Returns:
            DataFrame containing the loaded data.
        """

        import sqlalchemy

        engine = sqlalchemy.create_engine(connection)

        return cast(pd.DataFrame, pd.read_sql(query, engine, **kwargs))
