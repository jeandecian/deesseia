# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/jeandecian/deesseia/compare/) - YYYY-MM-DD

### Added

- None

### Changed

- None

### Deprecated

- None

### Removed

- None

### Fixed

- None

### Security

- None

## [1.0.0](https://github.com/jeandecian/deesseia/releases/tag/v1.0.0) - 2026-08-12

- **Added** `DataLoader`: Load data from CSV, JSON, Parquet, and SQL sources
- **Added** `DataInspector`: Inspect DataFrames with summary, head, tail, shape, columns, dtypes, and memory_usage methods
- **Added** `Cleaner`: Clean data with methods for handling missing values (mean, median, mode, ffill, bfill, drop), dropping duplicates, columns, empty columns, and single-cardinality columns
- **Added** `Validator`: Validate data with schema validation, missing value checks, and duplicate detection
- **Added** `FakeDataGenerator`: Generate synthetic numeric, categorical, mixed, and missing-value datasets for testing
- **Added** Comprehensive test suite with pytest and coverage reporting

## [1.1.0](https://github.com/jeandecian/deesseia/releases/tag/v1.1.0) - 2026-08-15

- **Added** `Scaler`: Feature scaling methods including MinMax, Standard, Robust, MaxAbs, and log transform
- **Added** `Encoder`: Categorical encoding methods including label, one-hot, target, frequency, and ordinal encoding
- **Added** `DataLoader.from_excel`: Load data from Excel files
