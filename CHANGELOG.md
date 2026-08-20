# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/jeandecian/deesseia/compare/) - YYYY-MM-DD

- **Added** None
- **Changed** None
- **Deprecated** None
- **Removed** None
- **Fixed** None
- **Security** None

## [1.4.1](https://github.com/jeandecian/deesseia/releases/tag/v1.4.1) - 2026-08-19

- **Added** `scipy>=1.10.0` to core dependencies
- **Added** `scipy.*` to mypy overrides with `ignore_missing_imports = true`
- **Added** `disable_error_code = ["import-untyped", "syntax"]` to mypy config for Python 3.12 compatibility
- **Fixed** `scipy-stubs` now conditionally installed only on Python 3.12+ (package requires Python 3.12+)
- **Fixed** `no-any-return` errors in `fake_data.py` and `encoder.py`
- **Fixed** `numpy.*` mypy override with `follow_imports = "skip"` to avoid Python 3.12+ syntax issues
- **Fixed** Mypy overrides syntax from `[tool.mypy.overrides]` to `[[tool.mypy.overrides]]` (array format required by mypy)

## [1.4.0](https://github.com/jeandecian/deesseia/releases/tag/v1.4.0) - 2026-08-19

- **Added** `DescriptiveStats`: Summary statistics including mean, median, mode, std, quartiles, IQR, skew, and kurtosis
- **Added** `Distributions`: Distribution analysis with histograms, KDE, boxplots, and violin plots
- **Added** `Insights`: Automated data insights including skew detection, outlier detection, high correlation detection, missing value analysis, and recommendations
- **Added** `Splitter.train_test_split`: Split data into train and test sets
- **Added** `Splitter.train_val_test_split`: Split data into train, validation, and test sets

## [1.3.0](https://github.com/jeandecian/deesseia/releases/tag/v1.3.0) - 2026-08-17

- **Added** `Splitter`: Cross-validation methods including K-Fold, Stratified K-Fold, and Time Series split

## [1.2.0](https://github.com/jeandecian/deesseia/releases/tag/v1.2.0) - 2026-08-15

- **Added** `Imputer`: Missing value imputation with strategies including mean, median, mode, constant, KNN, and model-based imputation
- **Added** `FeatureCreator`: Polynomial features and interaction features for feature engineering

## [1.1.0](https://github.com/jeandecian/deesseia/releases/tag/v1.1.0) - 2026-08-15

- **Added** `Scaler`: Feature scaling methods including MinMax, Standard, Robust, MaxAbs, and log transform
- **Added** `Encoder`: Categorical encoding methods including label, one-hot, target, frequency, and ordinal encoding
- **Added** `DataLoader.from_excel`: Load data from Excel files

## [1.0.0](https://github.com/jeandecian/deesseia/releases/tag/v1.0.0) - 2026-08-12

- **Added** `DataLoader`: Load data from CSV, JSON, Parquet, and SQL sources
- **Added** `DataInspector`: Inspect DataFrames with summary, head, tail, shape, columns, dtypes, and memory_usage methods
- **Added** `Cleaner`: Clean data with methods for handling missing values (mean, median, mode, ffill, bfill, drop), dropping duplicates, columns, empty columns, and single-cardinality columns
- **Added** `Validator`: Validate data with schema validation, missing value checks, and duplicate detection
- **Added** `FakeDataGenerator`: Generate synthetic numeric, categorical, mixed, and missing-value datasets for testing
- **Added** Comprehensive test suite with pytest and coverage reporting
