import pandas as pd
import pytest  # noqa: F401 # type: ignore[import-unused]

from deesseia.preprocess.encoder import Encoder


class TestEncoder:
    """Test feature encoding functionality."""

    def test_label_encode_default(self, sample_dataframe_for_encoding: pd.DataFrame):
        """Test label encoding on all object columns."""

        encoder: Encoder = Encoder()
        result: pd.DataFrame = encoder.label_encode(sample_dataframe_for_encoding)

        assert pd.api.types.is_numeric_dtype(result["category"])
        assert set(result["category"].unique()) == {0, 1, 2}
        assert pd.api.types.is_numeric_dtype(result["color"])
        assert set(result["color"].unique()) == {0, 1, 2}
        assert pd.api.types.is_numeric_dtype(result["size"])
        assert set(result["size"].unique()) == {0, 1, 2}

    def test_label_encode_with_columns(
        self, sample_dataframe_for_encoding: pd.DataFrame
    ):
        """Test label encoding on specific columns."""

        encoder: Encoder = Encoder()
        result: pd.DataFrame = encoder.label_encode(
            sample_dataframe_for_encoding, columns=["category"]
        )

        assert pd.api.types.is_numeric_dtype(result["category"])
        assert not pd.api.types.is_numeric_dtype(result["color"])

    def test_label_encode_nonexistent_column(
        self, sample_dataframe_for_encoding: pd.DataFrame
    ):
        """Test label encoding with a nonexistent column."""

        encoder: Encoder = Encoder()
        result: pd.DataFrame = encoder.label_encode(
            sample_dataframe_for_encoding, columns=["category", "nonexistent"]
        )

        assert pd.api.types.is_numeric_dtype(result["category"])
        assert "nonexistent" not in result.columns

    def test_one_hot_encode_default(self, sample_dataframe_for_encoding: pd.DataFrame):
        """Test one-hot encoding on all object columns."""

        encoder: Encoder = Encoder()
        result: pd.DataFrame = encoder.one_hot_encode(sample_dataframe_for_encoding)

        assert "category" not in result.columns
        assert "color" not in result.columns
        assert "size" not in result.columns
        assert "category_A" in result.columns
        assert "category_B" in result.columns
        assert "category_C" in result.columns
        assert "color_blue" in result.columns
        assert "color_green" in result.columns
        assert "color_red" in result.columns

    def test_one_hot_encode_drop_first(
        self, sample_dataframe_for_encoding: pd.DataFrame
    ):
        """Test one-hot encoding with drop_first=True."""

        encoder: Encoder = Encoder()
        result: pd.DataFrame = encoder.one_hot_encode(
            sample_dataframe_for_encoding, drop_first=True
        )

        assert "category_A" not in result.columns
        assert "category_B" in result.columns
        assert "category_C" in result.columns

    def test_one_hot_encode_with_columns(
        self, sample_dataframe_for_encoding: pd.DataFrame
    ):
        """Test one-hot encoding on specific columns."""

        encoder: Encoder = Encoder()
        result: pd.DataFrame = encoder.one_hot_encode(
            sample_dataframe_for_encoding, columns=["category"]
        )

        assert "category" not in result.columns
        assert "category_A" in result.columns
        assert "color" in result.columns

    def test_one_hot_encode_nonexistent_column(
        self, sample_dataframe_for_encoding: pd.DataFrame
    ):
        """Test one-hot encoding with a nonexistent column (should be skipped)."""

        encoder: Encoder = Encoder()
        result: pd.DataFrame = encoder.one_hot_encode(
            sample_dataframe_for_encoding, columns=["category", "nonexistent"]
        )

        assert "category" not in result.columns
        assert "category_A" in result.columns
        assert "nonexistent" not in result.columns

    def test_target_encode(self, sample_dataframe_for_encoding: pd.DataFrame):
        """Test target encoding."""

        encoder: Encoder = Encoder()
        result: pd.DataFrame = encoder.target_encode(
            sample_dataframe_for_encoding, column="category", target="target"
        )

        assert pd.api.types.is_numeric_dtype(result["category"])

    def test_target_encode_with_smoothing(
        self, sample_dataframe_for_encoding: pd.DataFrame
    ):
        """Test target encoding with smoothing."""

        encoder: Encoder = Encoder()
        result: pd.DataFrame = encoder.target_encode(
            sample_dataframe_for_encoding,
            column="category",
            target="target",
            smoothing=0.5,
        )

        assert pd.api.types.is_numeric_dtype(result["category"])

    def test_target_encode_missing_column(
        self, sample_dataframe_for_encoding: pd.DataFrame
    ):
        """Test target encoding with a missing column."""

        encoder: Encoder = Encoder()
        result: pd.DataFrame = encoder.target_encode(
            sample_dataframe_for_encoding, column="nonexistent", target="target"
        )

        assert result.equals(sample_dataframe_for_encoding)

    def test_target_encode_missing_target(
        self, sample_dataframe_for_encoding: pd.DataFrame
    ):
        """Test target encoding with a missing target column."""

        encoder: Encoder = Encoder()
        result: pd.DataFrame = encoder.target_encode(
            sample_dataframe_for_encoding, column="category", target="nonexistent"
        )

        assert result.equals(sample_dataframe_for_encoding)

    def test_target_encode_smoothing_edge_case(
        self, sample_dataframe_for_encoding: pd.DataFrame
    ):
        """Test target encoding when count < min_samples_leaf (smoothing branch)."""

        encoder: Encoder = Encoder()

        # Use min_samples_leaf > count to force smoothing branch
        result: pd.DataFrame = encoder.target_encode(
            sample_dataframe_for_encoding,
            column="category",
            target="target",
            min_samples_leaf=10,  # Larger than any category count
            smoothing=1.0,
        )

        category_means: pd.Series = sample_dataframe_for_encoding.groupby("category")[
            "target"
        ].mean()

        assert pd.api.types.is_numeric_dtype(result["category"])
        assert result["category"].iloc[0] != category_means.iloc[0]

    def test_frequency_encode_default(
        self, sample_dataframe_for_encoding: pd.DataFrame
    ):
        """Test frequency encoding (normalized)."""

        encoder: Encoder = Encoder()
        result: pd.DataFrame = encoder.frequency_encode(sample_dataframe_for_encoding)

        assert result["category"].min() >= 0
        assert result["category"].max() <= 1
        assert abs(result["category"].iloc[0] - 0.375) < 0.001  # A appears 3/8 times

    def test_frequency_encode_non_normalized(
        self, sample_dataframe_for_encoding: pd.DataFrame
    ):
        """Test frequency encoding without normalization."""

        encoder: Encoder = Encoder()
        result: pd.DataFrame = encoder.frequency_encode(
            sample_dataframe_for_encoding, normalize=False
        )

        assert result["category"].max() == 3  # A appears 3 times
        assert result["category"].min() == 2  # B and C appear 2-3 times

    def test_frequency_encode_with_columns(
        self, sample_dataframe_for_encoding: pd.DataFrame
    ):
        """Test frequency encoding on specific columns."""

        encoder: Encoder = Encoder()
        result: pd.DataFrame = encoder.frequency_encode(
            sample_dataframe_for_encoding, columns=["category"]
        )

        assert pd.api.types.is_numeric_dtype(result["category"])
        assert not pd.api.types.is_numeric_dtype(result["color"])

    def test_frequency_encode_nonexistent_column(
        self, sample_dataframe_for_encoding: pd.DataFrame
    ):
        """Test frequency encoding with a nonexistent column (should be skipped)."""

        encoder: Encoder = Encoder()
        result: pd.DataFrame = encoder.frequency_encode(
            sample_dataframe_for_encoding, columns=["category", "nonexistent"]
        )

        assert pd.api.types.is_numeric_dtype(result["category"])
        assert "nonexistent" not in result.columns

    def test_ordinal_encode_default(self, sample_dataframe_ordinal: pd.DataFrame):
        """Test ordinal encoding default (alphabetical order)."""

        encoder: Encoder = Encoder()
        result: pd.DataFrame = encoder.ordinal_encode(sample_dataframe_ordinal)

        assert pd.api.types.is_numeric_dtype(result["education"])
        assert pd.api.types.is_numeric_dtype(result["income"])

    def test_ordinal_encode_custom_ordering(
        self, sample_dataframe_ordinal: pd.DataFrame
    ):
        """Test ordinal encoding with custom ordering."""

        encoder: Encoder = Encoder()
        ordering = {
            "education": ["high_school", "bachelor", "master", "phd"],
            "income": ["low", "medium", "high"],
        }
        result: pd.DataFrame = encoder.ordinal_encode(
            sample_dataframe_ordinal, ordering=ordering
        )

        assert result["education"].iloc[0] == 0  # high_school
        assert result["education"].iloc[1] == 1  # bachelor

    def test_ordinal_encode_with_columns(self, sample_dataframe_ordinal: pd.DataFrame):
        """Test ordinal encoding on specific columns."""

        encoder: Encoder = Encoder()
        result: pd.DataFrame = encoder.ordinal_encode(
            sample_dataframe_ordinal, columns=["education"]
        )

        assert pd.api.types.is_numeric_dtype(result["education"])
        assert not pd.api.types.is_numeric_dtype(result["income"])

    def test_ordinal_encode_nonexistent_column(
        self, sample_dataframe_ordinal: pd.DataFrame
    ):
        """Test ordinal encoding with a nonexistent column (should be skipped)."""

        encoder: Encoder = Encoder()
        result: pd.DataFrame = encoder.ordinal_encode(
            sample_dataframe_ordinal, columns=["education", "nonexistent"]
        )

        assert pd.api.types.is_numeric_dtype(result["education"])
        assert "nonexistent" not in result.columns
