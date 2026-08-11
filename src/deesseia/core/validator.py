from __future__ import annotations

from typing import cast

import pandas as pd

from deesseia.core.base.validator import BaseValidator


class Validator(BaseValidator):
    """Validate data against schemas and check data quality."""

    def validate_schema(
        self,
        df: pd.DataFrame,
        schema: dict[str, str],
    ) -> dict[str, list[str]]:
        """Validate DataFrame against a schema.

        Args:
            df: DataFrame to validate.
            schema: Dictionary mapping column names to expected data types.

        Returns:
            Dictionary with 'missing_columns' and 'wrong_dtypes' lists.
        """

        errors: dict[str, list[str]] = {
            "missing_columns": [],
            "wrong_dtypes": [],
        }

        for col, expected_dtype in schema.items():
            if col not in df.columns:
                errors["missing_columns"].append(col)
            else:
                actual_dtype = str(df[col].dtype)
                if actual_dtype != expected_dtype:
                    errors["wrong_dtypes"].append(col)

        return errors

    def check_missing_values(self, df: pd.DataFrame) -> dict[str, float]:
        """Check for missing values in each column.

        Args:
            df: DataFrame to check.

        Returns:
            Dictionary mapping column names to missing percentage.
        """

        return cast(dict[str, float], (df.isnull().sum() / len(df) * 100).to_dict())

    def check_duplicates(self, df: pd.DataFrame) -> dict[str, int]:
        """Check for duplicate rows.

        Args:
            df: DataFrame to check.

        Returns:
            Dictionary with 'total_duplicates' count.
        """

        return {"total_duplicates": df.duplicated().sum()}
