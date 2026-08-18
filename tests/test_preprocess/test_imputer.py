import pandas as pd
import pytest  # noqa: F401 # type: ignore[import-unused]

from deesseia.preprocess.imputer import Imputer


class TestImputer:
    """Test missing value imputation functionality."""

    def test_mean_impute(self, sample_dataframe_with_missing: pd.DataFrame) -> None:
        """Test mean imputation."""

        imputer: Imputer = Imputer()
        result: pd.DataFrame = imputer.mean_impute(sample_dataframe_with_missing)

        assert result["col1"].iloc[2] == 3.0
        assert result["col2"].iloc[0] == 25.0

    def test_mean_impute_with_columns(
        self, sample_dataframe_with_missing: pd.DataFrame
    ) -> None:
        """Test mean imputation on specific columns."""

        imputer: Imputer = Imputer()
        result: pd.DataFrame = imputer.mean_impute(
            sample_dataframe_with_missing, columns=["col1"]
        )

        assert result["col1"].iloc[2] == 3.0
        assert result["col2"].isnull().any()

    def test_mean_impute_nonexistent_column(
        self, sample_dataframe_with_missing: pd.DataFrame
    ) -> None:
        """Test mean imputation with a nonexistent column."""

        imputer: Imputer = Imputer()
        result: pd.DataFrame = imputer.mean_impute(
            sample_dataframe_with_missing, columns=["col1", "nonexistent"]
        )

        assert result["col1"].iloc[2] == 3.0
        assert "nonexistent" not in result.columns

    def test_mean_impute_all_nan(self, sample_dataframe_all_nan: pd.DataFrame) -> None:
        """Test mean imputation with a column where all values are NaN."""

        imputer: Imputer = Imputer()
        result: pd.DataFrame = imputer.mean_impute(sample_dataframe_all_nan)

        assert result["all_nan"].isnull().all()
        assert result["normal"].iloc[0] == 1

    def test_median_impute(self, sample_dataframe_with_missing: pd.DataFrame) -> None:
        """Test median imputation."""

        imputer: Imputer = Imputer()
        result: pd.DataFrame = imputer.median_impute(sample_dataframe_with_missing)

        assert result["col1"].iloc[2] == 3.0
        assert result["col2"].iloc[0] == 25.0

    def test_median_impute_nonexistent_column(
        self, sample_dataframe_with_missing: pd.DataFrame
    ) -> None:
        """Test median imputation with a nonexistent column."""

        imputer: Imputer = Imputer()
        result: pd.DataFrame = imputer.median_impute(
            sample_dataframe_with_missing, columns=["col1", "nonexistent"]
        )

        assert result["col1"].iloc[2] == 3.0
        assert "nonexistent" not in result.columns

    def test_median_impute_all_nan(
        self, sample_dataframe_all_nan: pd.DataFrame
    ) -> None:
        """Test median imputation with a column where all values are NaN."""

        imputer: Imputer = Imputer()
        result: pd.DataFrame = imputer.median_impute(sample_dataframe_all_nan)

        assert result["all_nan"].isnull().all()
        assert result["normal"].iloc[0] == 1

    def test_mode_impute(self, sample_dataframe_with_missing: pd.DataFrame) -> None:
        """Test mode imputation."""

        imputer: Imputer = Imputer()
        result: pd.DataFrame = imputer.mode_impute(sample_dataframe_with_missing)

        assert result["col3"].iloc[3] == "a"

    def test_mode_impute_nonexistent_column(
        self, sample_dataframe_with_missing: pd.DataFrame
    ) -> None:
        """Test mode imputation with a nonexistent column."""

        imputer: Imputer = Imputer()
        result: pd.DataFrame = imputer.mode_impute(
            sample_dataframe_with_missing, columns=["col3", "nonexistent"]
        )

        assert result["col3"].iloc[3] == "a"
        assert "nonexistent" not in result.columns

    def test_constant_impute(self, sample_dataframe_with_missing: pd.DataFrame) -> None:
        """Test constant imputation."""

        imputer: Imputer = Imputer()
        result: pd.DataFrame = imputer.constant_impute(
            sample_dataframe_with_missing, value=-999
        )

        assert result["col1"].iloc[2] == -999
        assert result["col2"].iloc[0] == -999

    def test_constant_impute_with_columns(
        self, sample_dataframe_with_missing: pd.DataFrame
    ) -> None:
        """Test constant imputation on specific columns."""

        imputer: Imputer = Imputer()
        result: pd.DataFrame = imputer.constant_impute(
            sample_dataframe_with_missing, value=-999, columns=["col1"]
        )

        assert result["col1"].iloc[2] == -999
        assert result["col2"].isnull().any()

    def test_constant_impute_nonexistent_column(
        self, sample_dataframe_with_missing: pd.DataFrame
    ) -> None:
        """Test constant imputation with a nonexistent column."""

        imputer: Imputer = Imputer()
        result: pd.DataFrame = imputer.constant_impute(
            sample_dataframe_with_missing, value=-999, columns=["col1", "nonexistent"]
        )

        assert result["col1"].iloc[2] == -999
        assert "nonexistent" not in result.columns

    def test_knn_impute(self, sample_dataframe_for_imputation: pd.DataFrame) -> None:
        """Test KNN imputation."""

        imputer: Imputer = Imputer()
        result: pd.DataFrame = imputer.knn_impute(
            sample_dataframe_for_imputation, n_neighbors=2
        )

        assert not result.isnull().any().any()

    def test_knn_impute_no_numeric_columns(
        self, sample_dataframe_only_non_numeric: pd.DataFrame
    ) -> None:
        """Test KNN imputation with no numeric columns."""

        imputer: Imputer = Imputer()
        result: pd.DataFrame = imputer.knn_impute(sample_dataframe_only_non_numeric)

        assert result.equals(sample_dataframe_only_non_numeric)

    def test_knn_impute_columns_not_numeric(
        self, sample_dataframe_mixed_numeric_categorical: pd.DataFrame
    ) -> None:
        """Test KNN imputation with non-numeric columns specified."""

        imputer: Imputer = Imputer()
        result: pd.DataFrame = imputer.knn_impute(
            sample_dataframe_mixed_numeric_categorical, columns=["a", "b"]
        )

        assert result["a"].iloc[0] == "x"
        assert not result["b"].isnull().any()

    def test_knn_impute_columns_not_numeric_all(
        self, sample_dataframe_only_non_numeric: pd.DataFrame
    ) -> None:
        """Test KNN imputation with non-numeric columns specified."""

        imputer: Imputer = Imputer()
        result: pd.DataFrame = imputer.knn_impute(
            sample_dataframe_only_non_numeric, columns=["a", "b"]
        )

        assert result.equals(sample_dataframe_only_non_numeric)

    def test_model_impute(
        self, sample_dataframe_for_model_impute: pd.DataFrame
    ) -> None:
        """Test model-based imputation."""

        imputer: Imputer = Imputer()
        result: pd.DataFrame = imputer.model_impute(
            sample_dataframe_for_model_impute, target="target", features=["x1", "x2"]
        )

        assert not result["target"].isnull().any()
        assert abs(result["target"].iloc[2] - 300) < 10

    def test_model_impute_default_features(
        self, sample_dataframe_for_model_impute: pd.DataFrame
    ) -> None:
        """Test model imputation with default features (all numeric except target)."""

        imputer: Imputer = Imputer()
        result: pd.DataFrame = imputer.model_impute(
            sample_dataframe_for_model_impute, target="target"
        )

        assert not result["target"].isnull().any()
        assert abs(result["target"].iloc[2] - 300) < 10

    def test_model_impute_missing_target_returns_original(
        self, sample_dataframe_with_missing: pd.DataFrame
    ) -> None:
        """Test model imputation with a missing target column returns original."""

        imputer: Imputer = Imputer()
        result: pd.DataFrame = imputer.model_impute(
            sample_dataframe_with_missing, target="nonexistent"
        )

        assert result.equals(sample_dataframe_with_missing)

    def test_model_impute_no_features(
        self, sample_dataframe_single_target: pd.DataFrame
    ) -> None:
        """Test model imputation with no features."""

        imputer: Imputer = Imputer()
        result: pd.DataFrame = imputer.model_impute(
            sample_dataframe_single_target, target="target", features=[]
        )

        assert result.equals(sample_dataframe_single_target)

    def test_model_impute_train_data_empty(
        self, sample_dataframe_all_nan: pd.DataFrame
    ) -> None:
        """Test model imputation when training data is empty."""

        imputer: Imputer = Imputer()
        result: pd.DataFrame = imputer.model_impute(
            sample_dataframe_all_nan, target="target"
        )

        assert result.equals(sample_dataframe_all_nan)

    def test_model_impute_no_data(
        self, sample_dataframe_all_nan_model: pd.DataFrame
    ) -> None:
        """Test model imputation when both train and predict data are empty."""

        imputer: Imputer = Imputer()
        result: pd.DataFrame = imputer.model_impute(
            sample_dataframe_all_nan_model, target="target"
        )

        assert result.equals(sample_dataframe_all_nan_model)

    def test_model_impute_predict_data_empty(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test model imputation when prediction data is empty (no missing values)."""

        imputer: Imputer = Imputer()
        result: pd.DataFrame = imputer.model_impute(sample_dataframe, target="target")

        assert result.equals(sample_dataframe)

    def test_model_impute_custom_model(
        self, sample_dataframe_for_model_impute: pd.DataFrame
    ) -> None:
        """Test model imputation with custom model."""

        from sklearn.linear_model import Ridge

        imputer: Imputer = Imputer()
        result: pd.DataFrame = imputer.model_impute(
            sample_dataframe_for_model_impute,
            target="target",
            features=["x1", "x2"],
            model=Ridge(alpha=1.0),
        )

        assert not result["target"].isnull().any()
