import pandas as pd
import pytest  # noqa: F401 # type: ignore[import-unused]

from deesseia.core.feature import FeatureCreator


class TestFeatureCreator:
    """Test feature creation functionality."""

    def test_polynomial_features_default(
        self, sample_dataframe_poly: pd.DataFrame
    ) -> None:
        """Test polynomial features with default parameters."""

        creator: FeatureCreator = FeatureCreator()
        result: pd.DataFrame = creator.polynomial_features(
            sample_dataframe_poly, degree=2
        )

        assert "x" in result.columns
        assert "y" in result.columns
        assert "x^2" in result.columns
        assert "y^2" in result.columns
        assert result["x^2"].iloc[0] == 1
        assert result["x^2"].iloc[1] == 4

    def test_polynomial_features_degree_3(
        self, sample_dataframe_single_col: pd.DataFrame
    ) -> None:
        """Test polynomial features with degree 3."""

        creator: FeatureCreator = FeatureCreator()
        result: pd.DataFrame = creator.polynomial_features(
            sample_dataframe_single_col, degree=3
        )

        assert "x" in result.columns
        assert "x^2" in result.columns
        assert "x^3" in result.columns
        assert result["x^3"].iloc[0] == 1
        assert result["x^3"].iloc[1] == 8

    def test_polynomial_features_with_columns(
        self, sample_dataframe_poly_three: pd.DataFrame
    ) -> None:
        """Test polynomial features on specific columns."""

        creator: FeatureCreator = FeatureCreator()
        result: pd.DataFrame = creator.polynomial_features(
            sample_dataframe_poly_three, columns=["x", "y"], degree=2
        )

        assert "x" in result.columns
        assert "y" in result.columns
        assert "z" in result.columns
        assert "x^2" in result.columns
        assert "y^2" in result.columns
        assert "z^2" not in result.columns

    def test_polynomial_features_with_bias(
        self, sample_dataframe_single_col: pd.DataFrame
    ) -> None:
        """Test polynomial features with bias term."""

        creator: FeatureCreator = FeatureCreator()
        result: pd.DataFrame = creator.polynomial_features(
            sample_dataframe_single_col, include_bias=True
        )

        assert "bias" in result.columns
        assert (result["bias"] == 1.0).all()

    def test_polynomial_features_no_columns(
        self, sample_dataframe_non_numeric: pd.DataFrame
    ) -> None:
        """Test polynomial features with no numeric columns."""

        creator: FeatureCreator = FeatureCreator()
        result: pd.DataFrame = creator.polynomial_features(sample_dataframe_non_numeric)

        assert result.equals(sample_dataframe_non_numeric)

    def test_polynomial_features_nonexistent_column(
        self, sample_dataframe_single_col: pd.DataFrame
    ) -> None:
        """Test polynomial features with a nonexistent column."""

        creator: FeatureCreator = FeatureCreator()
        result: pd.DataFrame = creator.polynomial_features(
            sample_dataframe_single_col, columns=["x", "nonexistent"]
        )

        assert "x" in result.columns
        assert "x^2" in result.columns
        assert "nonexistent" not in result.columns

    def test_interaction_features_default(
        self, sample_dataframe_poly: pd.DataFrame
    ) -> None:
        """Test interaction features with default parameters."""

        creator: FeatureCreator = FeatureCreator()
        result: pd.DataFrame = creator.interaction_features(sample_dataframe_poly)

        assert "x" in result.columns
        assert "y" in result.columns
        assert "x_y" in result.columns
        assert result["x_y"].iloc[0] == 2
        assert result["x_y"].iloc[1] == 8

    def test_interaction_features_multiple_columns(
        self, sample_dataframe_poly_three: pd.DataFrame
    ) -> None:
        """Test interaction features with multiple columns."""

        creator: FeatureCreator = FeatureCreator()
        result: pd.DataFrame = creator.interaction_features(sample_dataframe_poly_three)

        assert "x_y" in result.columns
        assert "x_z" in result.columns
        assert "y_z" in result.columns

    def test_interaction_features_single_column(
        self, sample_dataframe_single_col: pd.DataFrame
    ) -> None:
        """Test interaction features with a single column."""

        creator: FeatureCreator = FeatureCreator()
        result: pd.DataFrame = creator.interaction_features(sample_dataframe_single_col)

        assert result.equals(sample_dataframe_single_col)

    def test_interaction_features_with_columns(
        self, sample_dataframe_poly_three: pd.DataFrame
    ) -> None:
        """Test interaction features on specific columns."""

        creator: FeatureCreator = FeatureCreator()
        result: pd.DataFrame = creator.interaction_features(
            sample_dataframe_poly_three, columns=["x", "y"]
        )

        assert "x_y" in result.columns
        assert "x_z" not in result.columns
        assert "y_z" not in result.columns

    def test_interaction_features_nonexistent_column(
        self, sample_dataframe_poly_three: pd.DataFrame
    ) -> None:
        """Test interaction features with a nonexistent column."""

        creator: FeatureCreator = FeatureCreator()
        result: pd.DataFrame = creator.interaction_features(
            sample_dataframe_poly_three, columns=["x", "nonexistent"]
        )

        assert "x" in result.columns
        assert "y" in result.columns
        assert "z" in result.columns
        assert "x_y" not in result.columns
        assert "x_z" not in result.columns
        assert "y_z" not in result.columns
        assert "nonexistent" not in result.columns

    def test_polynomial_and_interaction_features(
        self, sample_dataframe_poly: pd.DataFrame
    ) -> None:
        """Test both polynomial and interaction features together."""

        creator: FeatureCreator = FeatureCreator()
        result: pd.DataFrame = creator.polynomial_and_interaction_features(
            sample_dataframe_poly, degree=2
        )

        assert "x" in result.columns
        assert "y" in result.columns
        assert "x^2" in result.columns
        assert "y^2" in result.columns
        assert "x_y" in result.columns

    def test_polynomial_and_interaction_features_with_bias(
        self, sample_dataframe_poly: pd.DataFrame
    ) -> None:
        """Test both polynomial and interaction features with bias."""

        creator: FeatureCreator = FeatureCreator()
        result: pd.DataFrame = creator.polynomial_and_interaction_features(
            sample_dataframe_poly, include_bias=True
        )

        assert "bias" in result.columns
        assert (result["bias"] == 1.0).all()

    def test_polynomial_and_interaction_features_no_columns(
        self, sample_dataframe_non_numeric: pd.DataFrame
    ) -> None:
        """Test both polynomial and interaction features with no numeric columns."""

        creator: FeatureCreator = FeatureCreator()
        result: pd.DataFrame = creator.polynomial_and_interaction_features(
            sample_dataframe_non_numeric
        )

        assert result.equals(sample_dataframe_non_numeric)

    def test_polynomial_and_interaction_features_nonexistent_column(
        self, sample_dataframe_poly: pd.DataFrame
    ) -> None:
        """Test polynomial and interaction features with a nonexistent column."""

        creator: FeatureCreator = FeatureCreator()
        result: pd.DataFrame = creator.polynomial_and_interaction_features(
            sample_dataframe_poly, columns=["x", "nonexistent", "y"], degree=2
        )

        assert "x" in result.columns
        assert "y" in result.columns
        assert "x^2" in result.columns
        assert "y^2" in result.columns
        assert "x_y" in result.columns
        assert "nonexistent" not in result.columns

    def test_polynomial_and_interaction_features_missing_col_in_interaction(
        self, sample_dataframe_poly_three: pd.DataFrame
    ) -> None:
        """Test polynomial and interaction features when a column is missing from interaction."""

        creator: FeatureCreator = FeatureCreator()
        result: pd.DataFrame = creator.polynomial_and_interaction_features(
            sample_dataframe_poly_three, columns=["x", "y", "nonexistent"], degree=2
        )

        assert "x" in result.columns
        assert "y" in result.columns
        assert "z" in result.columns
        assert "x^2" in result.columns
        assert "y^2" in result.columns
        assert "z^2" not in result.columns
        assert "x_y" in result.columns
        assert "x_z" not in result.columns
        assert "y_z" not in result.columns
