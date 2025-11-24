# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python script for analyzing temperature trends from Home Assistant historical data. The script performs comprehensive statistical analysis including trend detection, degree days calculation, temporal patterns, and anomaly detection.

## Running the Script

The script supports two input methods:

### From Home Assistant API (Primary Method)
```bash
python ha-temp-trend.py \
  --ha-sensor sensor.patio_climate_temperature \
  --ha-url http://homeassistant.local:8123 \
  --ha-token YOUR_LONG_LIVED_TOKEN
```

Optional date range parameters:
- `--start-date YYYY-MM-DD` (defaults to 30 days ago)
- `--end-date YYYY-MM-DD` (defaults to today)

### From CSV File (Alternative Method)
```bash
python ha-temp-trend.py --csv-file path/to/data.csv
```

CSV must contain columns: `entity_id`, `state`, `last_changed`

### Environment Variables
The Home Assistant token can be set via the `HA_TOKEN` environment variable instead of the `--ha-token` parameter.

## Dependencies

Required Python packages (standard library + third-party):
- pandas
- numpy
- scipy
- requests
- argparse (standard library)
- datetime (standard library)
- os (standard library)
- sys (standard library)

Install third-party dependencies:
```bash
pip install pandas numpy scipy requests
```

## Architecture

### Main Components

**`fetch_data_from_home_assistant()`** (ha-temp-trend.py:10-85)
- Fetches historical data from Home Assistant's REST API
- Handles timezone conversion (local to UTC for API calls)
- Uses the `/api/history/period/{start_iso}` endpoint with minimal_response mode
- Returns DataFrame with entity_id, state, and last_changed columns

**`calculate_temperature_statistics()`** (ha-temp-trend.py:87-386)
- Core analysis function that processes temperature data
- Performs all statistical calculations and outputs results
- Uses local timezone for all displayed timestamps and time-based groupings

### Analysis Pipeline

1. **Data Preprocessing** (lines 109-142)
   - Filters out invalid states (unavailable/unknown/None/null)
   - Converts to numeric values, dropping NaN
   - Converts all timestamps to local timezone for consistent analysis
   - Sorts by timestamp and calculates hours from start

2. **Statistical Calculations** (lines 144-199)
   - Basic statistics: mean, median, std dev, min/max
   - Moving average (3-period window)
   - Trend analysis: identifies rising/cooling trends using temperature differences
   - Linear regression for each trend group (calculates R-squared values)

3. **Temporal Analysis** (lines 200-241)
   - Resamples to hourly means for consistent time-based analysis
   - Heating/Cooling Degree Days (baseline 65°F by default)
   - Hour-of-day and day-of-week patterns
   - Diurnal Temperature Range (DTR) calculation
   - Autocorrelation at lag-1 and lag-24

4. **Anomaly Detection** (lines 242-288)
   - Heat spells: consecutive hours above 80°F
   - Cold spells: consecutive hours below 40°F
   - Abrupt changes: uses 3-sigma outlier detection on hourly temperature differences

### Timezone Handling

Critical aspect of the codebase:
- Input datetimes from argparse are assumed to be local time and converted to UTC for API calls
- Home Assistant returns UTC timestamps
- All timestamps are converted to local timezone (ha-temp-trend.py:130) for processing and display
- This ensures consistent analysis and user-friendly output

### Key Design Patterns

- **Trend Detection**: Uses cumulative sum of boolean mask where trend changes direction to create group IDs for consecutive rising/cooling periods
- **Spell Detection**: Similar pattern for heat/cold spells - cumulative sum on boolean shift changes creates spell IDs
- **Data Cleaning**: Creates copy of DataFrame after filtering (line 116) to avoid SettingWithCopyWarning

## Common Modifications

When modifying thresholds or baseline temperatures:
- Heat spell threshold: line 243 (default 80°F)
- Cold spell threshold: line 244 (default 40°F)
- Degree days baseline: `baseline_temp` parameter in `calculate_temperature_statistics()` (default 65°F)
- Abrupt change detection: line 285 (3-sigma threshold)

When adding new temporal patterns:
- Work with `hourly_df` DataFrame created at line 203
- Timestamps are already in local timezone and resampled to hourly means
- Can group by additional temporal fields (week, month, etc.)

When modifying Home Assistant API calls:
- Endpoint structure: ha-temp-trend.py:39
- Minimal response mode used for efficiency: line 48
- Timeout set to 30 seconds: line 54
