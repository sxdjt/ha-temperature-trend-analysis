# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-03-30

### Added
- .gitignore for Python, virtual environments, IDE files, and OS artifacts

### Fixed
- Removed stray character from README.md sample output

## [1.0.0] - 2026-02-03

### Added
- Initial release of ha-temperature-trend-analysis
- Fetch historical temperature data from Home Assistant REST API or CSV file
- Statistical analysis: mean, median, standard deviation, min/max
- Trend detection with linear regression and R-squared values
- Heating and Cooling Degree Days (HDD/CDD) calculation
- Hour-of-day and day-of-week temporal patterns
- Diurnal Temperature Range (DTR) calculation
- Autocorrelation analysis at lag-1 and lag-24
- Heat spell and cold spell detection
- Abrupt change detection using 3-sigma outlier method
- Timezone-aware analysis (UTC API data converted to local time)
- Support for long-lived Home Assistant token via CLI or environment variable
