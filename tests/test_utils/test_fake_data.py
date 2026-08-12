import pandas as pd
import pytest

from deesseia.utils.fake_data import FakeDataGenerator


class TestFakeDataGenerator:
    """Test fake data generation functionality."""

    def test_init(self):
        """Test initialization with and without seed."""

        gen1: FakeDataGenerator = FakeDataGenerator(random_state=42)
        gen2: FakeDataGenerator = FakeDataGenerator(random_state=42)

        assert gen1.rng is not None
        assert gen2.rng is not None

    def test_generate_numeric_normal(self, fake_data_generator: FakeDataGenerator):
        """Test generating normal distribution data."""

        df: pd.DataFrame = fake_data_generator.generate_numeric(
            n_samples=100, n_features=5
        )

        assert df.shape == (100, 5)
        assert all(df.dtypes == "float64")

    def test_generate_numeric_uniform(self, fake_data_generator: FakeDataGenerator):
        """Test generating uniform distribution data."""

        df: pd.DataFrame = fake_data_generator.generate_numeric(
            n_samples=100, n_features=3, distribution="uniform"
        )

        assert df.shape == (100, 3)
        assert df.min().min() >= 0
        assert df.max().max() <= 1

    def test_generate_numeric_exponential(self, fake_data_generator: FakeDataGenerator):
        """Test generating exponential distribution data."""

        df: pd.DataFrame = fake_data_generator.generate_numeric(
            n_samples=100, n_features=3, distribution="exponential"
        )

        assert df.shape == (100, 3)
        assert (df >= 0).all().all()

    def test_generate_numeric_poisson(self, fake_data_generator: FakeDataGenerator):
        """Test generating Poisson distribution data."""

        df: pd.DataFrame = fake_data_generator.generate_numeric(
            n_samples=100, n_features=3, distribution="poisson"
        )

        assert df.shape == (100, 3)
        assert (df >= 0).all().all()
        assert all(df.dtypes == "int64")

    def test_generate_numeric_unsupported_distribution(
        self, fake_data_generator: FakeDataGenerator
    ):
        """Test unsupported distribution raises error."""

        with pytest.raises(ValueError, match="Unsupported distribution: invalid"):
            fake_data_generator.generate_numeric(distribution="invalid")

    def test_generate_categorical_default(self, fake_data_generator: FakeDataGenerator):
        """Test generating categorical data with default categories."""

        df: pd.DataFrame = fake_data_generator.generate_categorical(
            n_samples=100, n_features=3
        )
        valid_categories: list[str] = ["A", "B", "C", "D"]

        assert df.shape == (100, 3)
        assert all(df.iloc[:, 0].isin(valid_categories))

    def test_generate_categorical_custom(self, fake_data_generator: FakeDataGenerator):
        """Test generating categorical data with custom categories."""

        categories: list[str] = ["X", "Y", "Z"]
        df: pd.DataFrame = fake_data_generator.generate_categorical(
            n_samples=50, n_features=2, categories=categories
        )

        assert df.shape == (50, 2)
        assert all(df.iloc[:, 0].isin(categories))

    def test_generate_mixed(self, fake_data_generator: FakeDataGenerator):
        """Test generating mixed numeric and categorical data."""

        df: pd.DataFrame = fake_data_generator.generate_mixed(
            n_samples=100, n_numeric=3, n_categorical=2
        )

        assert df.shape == (100, 5)
        assert "feature_1" in df.columns
        assert "feature_2" in df.columns
        assert "feature_3" in df.columns
        assert "category_1" in df.columns
        assert "category_2" in df.columns

    def test_generate_missing_data(self, fake_data_generator: FakeDataGenerator):
        """Test generating data with missing values."""

        df: pd.DataFrame = fake_data_generator.generate_missing_data(
            n_rows=100, n_cols=5, missing_probability=0.1
        )

        assert df.shape == (100, 5)
        # Should have some missing values
        assert df.isnull().sum().sum() > 0

    def test_generate_missing_data_zero_probability(
        self, fake_data_generator: FakeDataGenerator
    ):
        """Test generating data with no missing values."""

        df: pd.DataFrame = fake_data_generator.generate_missing_data(
            n_rows=100, n_cols=5, missing_probability=0.0
        )

        assert df.isnull().sum().sum() == 0

    def test_generate_missing_data_full_probability(
        self, fake_data_generator: FakeDataGenerator
    ):
        """Test generating data with all missing values."""

        df: pd.DataFrame = fake_data_generator.generate_missing_data(
            n_rows=100, n_cols=5, missing_probability=1.0
        )

        assert df.isnull().sum().sum() == 500  # 100 * 5
