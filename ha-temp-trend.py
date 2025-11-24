"""Temperature trend analysis tool for Home Assistant historical data."""

# Standard library
import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

# Third-party
import numpy as np
import pandas as pd
import requests
from scipy import stats

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Temperature thresholds (°F)
HEAT_THRESHOLD_F = 80
COLD_THRESHOLD_F = 40
BASELINE_TEMP_F = 65

# Analysis parameters
MOVING_AVG_WINDOW = 3
OUTLIER_SIGMA_MULTIPLIER = 3

# API configuration
API_REQUEST_TIMEOUT = 30
DEFAULT_LOOKBACK_DAYS = 30

# Display formatting
SECTION_SEPARATOR = "=" * 60
SUBSECTION_SEPARATOR = "-" * 60


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SpellMetrics:
    """Metrics for temperature spell analysis (heat or cold)."""
    num_spells: int
    avg_duration: float
    avg_intensity: float


@dataclass
class TrendMetrics:
    """Metrics for rising and cooling trends."""
    avg_rate: float
    avg_duration: float
    avg_range: float
    avg_r_squared: float


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_start_date(date_str: str) -> datetime:
    """Parse start date string to datetime at beginning of day.

    Args:
        date_str: Date string in YYYY-MM-DD format.

    Returns:
        Datetime object at 00:00:00 of the specified date.
    """
    return datetime.strptime(date_str, '%Y-%m-%d').replace(
        hour=0, minute=0, second=0, microsecond=0
    )


def parse_end_date(date_str: str) -> datetime:
    """Parse end date string to datetime at end of day.

    Args:
        date_str: Date string in YYYY-MM-DD format.

    Returns:
        Datetime object at 23:59:59 of the specified date.
    """
    return datetime.strptime(date_str, '%Y-%m-%d').replace(
        hour=23, minute=59, second=59, microsecond=999999
    )


def format_spell_metric(metric_name: str, value: float) -> str:
    """Format spell metric values based on metric type.

    Args:
        metric_name: Name of the metric (e.g., "Average Duration").
        value: Numeric value to format.

    Returns:
        Formatted string with appropriate units.
    """
    if "Duration" in metric_name:
        return f"{value:.2f} hours"
    elif "Intensity" in metric_name:
        return f"{value:.2f}°F"
    else:
        return f"{int(value)}"


# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_data_from_home_assistant(
    ha_url: str,
    ha_token: str,
    sensor_name: str,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """Fetches historical temperature data from Home Assistant.

    Args:
        ha_url: The base URL of your Home Assistant instance
                (e.g., http://homeassistant.local:8123).
        ha_token: Your Home Assistant Long-Lived Access Token.
        sensor_name: The entity_id of the temperature sensor
                     (e.g., "sensor.home_temperature").
        start_date: The start datetime for fetching data (will be converted to UTC).
        end_date: The end datetime for fetching data (will be converted to UTC).

    Returns:
        DataFrame containing 'entity_id', 'state', and 'last_changed' columns,
        or an empty DataFrame if no data is found or an error occurs.
    """
    # Ensure dates are timezone-aware UTC for HA API
    if start_date.tzinfo is None:
        start_date = start_date.astimezone(timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.astimezone(timezone.utc)

    # HA API expects ISO 8601 format with 'Z' for UTC
    start_iso = start_date.isoformat(timespec='seconds').replace('+00:00', 'Z')
    end_iso = end_date.isoformat(timespec='seconds').replace('+00:00', 'Z')

    # Endpoint for history period
    api_url = f"{ha_url.rstrip('/')}/api/history/period/{start_iso}"

    headers = {
        "Authorization": f"Bearer {ha_token}",
        "Content-Type": "application/json",
    }
    params = {
        "filter_entity_id": sensor_name,
        "end_time": end_iso,
        "minimal_response": "true"
    }

    print(f"\n* Fetching data from Home Assistant for '{sensor_name}' "
          f"from {start_date.strftime('%Y-%m-%d %H:%M')} "
          f"to {end_date.strftime('%Y-%m-%d %H:%M')} (UTC)...")

    try:
        response = requests.get(
            api_url, headers=headers, params=params, timeout=API_REQUEST_TIMEOUT
        )
        response.raise_for_status()

        data = response.json()

        if not data or not isinstance(data, list) or not data[0]:
            print(f"\n\n*** No data found for sensor '{sensor_name}' "
                  f"in the specified period from Home Assistant.\n\n")
            return pd.DataFrame()

        # HA returns a list of lists (grouped by entity_id, even if just one)
        sensor_states = data[0]

        extracted_data = [
            {
                'entity_id': sensor_name,
                'state': state_obj.get('state'),
                'last_changed': state_obj.get('last_changed')
            }
            for state_obj in sensor_states
        ]

        df = pd.DataFrame(extracted_data)
        print(f"* Successfully fetched {len(df)} records from Home Assistant.")
        return df

    except requests.exceptions.RequestException as e:
        print(f"\n\n*** Error connecting to Home Assistant: {e}")
        print("*** Please ensure the URL is correct, Home Assistant is running, "
              "and your token is valid.\n\n")
        return pd.DataFrame()
    except Exception as e:
        print(f"\n\n*** An unexpected error occurred while fetching HA data: {e}\n\n")
        return pd.DataFrame()


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def preprocess_temperature_data(df: pd.DataFrame, local_tz) -> Optional[pd.DataFrame]:
    """Preprocess temperature data for analysis.

    Args:
        df: Raw DataFrame with entity_id, state, last_changed columns.
        local_tz: Local timezone for timestamp conversion.

    Returns:
        Cleaned DataFrame with temperature and timestamp columns, or None if empty.
    """
    required_columns = ['entity_id', 'state', 'last_changed']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(
            f"\n\n*** The input DataFrame must contain "
            f"{', '.join(required_columns)} columns.\n\n"
        )

    initial_rows = len(df)
    df = df[~df['state'].isin(['unavailable', 'unknown', 'None', 'null'])].copy()
    df['state'] = pd.to_numeric(df['state'], errors='coerce')
    df = df.dropna(subset=['state'])
    print(f"* Filtered out {initial_rows - len(df)} "
          f"non-numeric/unavailable temperature readings.")

    if df.empty:
        print("\n\n*** No valid temperature data remaining after cleaning. "
              "Exiting analysis.\n\n")
        return None

    df['temperature'] = df['state'].astype(float)
    df['timestamp'] = pd.to_datetime(df['last_changed'], format='ISO8601')

    # Ensure timestamps are timezone-aware and convert to local timezone
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(timezone.utc)
    df['timestamp'] = df['timestamp'].dt.tz_convert(local_tz)

    df = df.sort_values(by='timestamp').reset_index(drop=True)
    df['hours'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / 3600

    return df


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def calculate_basic_statistics(df: pd.DataFrame) -> dict:
    """Calculate basic temperature statistics.

    Args:
        df: DataFrame with temperature column.

    Returns:
        Dictionary of basic statistics.
    """
    return {
        'mean': df['temperature'].mean(),
        'median': df['temperature'].median(),
        'std_dev': df['temperature'].std(),
        'max': df['temperature'].max(),
        'min': df['temperature'].min()
    }


def analyze_trends(df: pd.DataFrame) -> tuple[TrendMetrics, TrendMetrics]:
    """Analyze rising and cooling temperature trends.

    Args:
        df: DataFrame with temperature, hours, and trend columns.

    Returns:
        Tuple of (rising_metrics, cooling_metrics).
    """
    df['temp_diff'] = df['temperature'].diff()
    df['trend'] = df['temp_diff'].apply(
        lambda x: 'rising' if x > 0 else ('cooling' if x < 0 else None)
    )
    df_trends = df.dropna(subset=['trend']).copy()

    trend_groups = df_trends.groupby(
        (df_trends['trend'] != df_trends['trend'].shift()).cumsum()
    )

    rise_rate, cool_rate = [], []
    rise_durations, cool_durations = [], []
    rise_ranges, cool_ranges = [], []
    rise_r_squared, cool_r_squared = [], []

    for _, group in trend_groups:
        if len(group) < 2:
            continue

        total_temp_change = group['temperature'].iloc[-1] - group['temperature'].iloc[0]
        total_hours = group['hours'].iloc[-1] - group['hours'].iloc[0]

        if total_hours == 0 or total_temp_change == 0:
            continue

        temp_range = group['temperature'].max() - group['temperature'].min()
        _, _, r_value, _, _ = stats.linregress(group['hours'], group['temperature'])
        r_squared = r_value ** 2

        if group['trend'].iloc[0] == 'rising':
            rise_rate.append(total_temp_change / total_hours)
            rise_durations.append(total_hours)
            rise_ranges.append(temp_range)
            rise_r_squared.append(r_squared)
        elif group['trend'].iloc[0] == 'cooling':
            cool_rate.append(total_temp_change / total_hours)
            cool_durations.append(total_hours)
            cool_ranges.append(temp_range)
            cool_r_squared.append(r_squared)

    rising_metrics = TrendMetrics(
        avg_rate=np.mean(rise_rate) if rise_rate else 0.0,
        avg_duration=np.mean(rise_durations) if rise_durations else 0.0,
        avg_range=np.mean(rise_ranges) if rise_ranges else 0.0,
        avg_r_squared=np.mean(rise_r_squared) if rise_r_squared else 0.0
    )

    cooling_metrics = TrendMetrics(
        avg_rate=np.mean(cool_rate) if cool_rate else 0.0,
        avg_duration=np.mean(cool_durations) if cool_durations else 0.0,
        avg_range=np.mean(cool_ranges) if cool_ranges else 0.0,
        avg_r_squared=np.mean(cool_r_squared) if cool_r_squared else 0.0
    )

    return rising_metrics, cooling_metrics


def prepare_hourly_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Resample data to hourly means for temporal analysis.

    Args:
        df: DataFrame with timestamp and temperature columns.

    Returns:
        Hourly resampled DataFrame or None if empty.
    """
    hourly_df = df.set_index('timestamp')['temperature'].resample('h').mean().to_frame()
    hourly_df.columns = ['temperature']
    hourly_df = hourly_df.dropna()

    if hourly_df.empty:
        print("Not enough hourly data to perform detailed daily/hourly analysis. "
              "Some sections will be skipped.")
        return None

    hourly_df['date'] = hourly_df.index.date
    hourly_df['day_of_week'] = hourly_df.index.day_name()
    hourly_df['hour_of_day'] = hourly_df.index.hour

    return hourly_df


def calculate_degree_days(
    hourly_df: pd.DataFrame,
    baseline_temp: float
) -> tuple[float, float]:
    """Calculate heating and cooling degree days.

    Args:
        hourly_df: Hourly temperature DataFrame.
        baseline_temp: Baseline temperature in °F.

    Returns:
        Tuple of (heating_degree_days, cooling_degree_days).
    """
    daily_avg_temp = hourly_df.groupby('date')['temperature'].mean()

    heating_dd = (baseline_temp - daily_avg_temp).apply(lambda x: max(0, x)).sum()
    cooling_dd = (daily_avg_temp - baseline_temp).apply(lambda x: max(0, x)).sum()

    return heating_dd, cooling_dd


def analyze_temporal_patterns(hourly_df: pd.DataFrame) -> dict:
    """Analyze temporal patterns in temperature data.

    Args:
        hourly_df: Hourly temperature DataFrame.

    Returns:
        Dictionary containing temporal analysis results.
    """
    avg_temp_by_hour = hourly_df.groupby('hour_of_day')['temperature'].mean()

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    avg_temp_by_day_of_week = hourly_df.groupby(
        pd.Categorical(hourly_df['day_of_week'], categories=day_order, ordered=True),
        observed=False
    )['temperature'].mean()

    # Diurnal Temperature Range (DTR)
    daily_max_temp = hourly_df.groupby('date')['temperature'].max()
    daily_min_temp = hourly_df.groupby('date')['temperature'].min()
    daily_dtr = daily_max_temp - daily_min_temp
    avg_daily_dtr = daily_dtr.mean() if not daily_dtr.empty else 0.0
    max_daily_dtr = daily_dtr.max() if not daily_dtr.empty else 0.0

    return {
        'avg_temp_by_hour': avg_temp_by_hour,
        'avg_temp_by_day_of_week': avg_temp_by_day_of_week,
        'avg_daily_dtr': avg_daily_dtr,
        'max_daily_dtr': max_daily_dtr
    }


def calculate_spell_metrics(
    hourly_df: pd.DataFrame,
    threshold: float,
    is_above: bool
) -> SpellMetrics:
    """Calculate spell metrics for temperature threshold exceedances.

    Args:
        hourly_df: Hourly temperature DataFrame.
        threshold: Temperature threshold in °F.
        is_above: True for heat spells (above threshold), False for cold spells.

    Returns:
        SpellMetrics object with spell analysis results.
    """
    if is_above:
        hourly_df['is_spell'] = hourly_df['temperature'] > threshold
    else:
        hourly_df['is_spell'] = hourly_df['temperature'] < threshold

    hourly_df['spell_id'] = (
        hourly_df['is_spell'] != hourly_df['is_spell'].shift(1)
    ).cumsum()

    spells_data = hourly_df[hourly_df['is_spell']]
    spells_grouped = spells_data.groupby('spell_id')

    num_spells = len(spells_grouped) if not spells_data.empty else 0
    avg_duration = 0.0
    avg_intensity = 0.0

    if num_spells > 0:
        spell_durations = [len(group) for _, group in spells_grouped]

        if is_above:
            spell_intensities = [
                group['temperature'].mean() - threshold
                for _, group in spells_grouped
            ]
        else:
            spell_intensities = [
                threshold - group['temperature'].mean()
                for _, group in spells_grouped
            ]

        avg_duration = np.mean(spell_durations)
        avg_intensity = np.mean(spell_intensities)

    return SpellMetrics(
        num_spells=num_spells,
        avg_duration=avg_duration,
        avg_intensity=avg_intensity
    )


def detect_abrupt_changes(hourly_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Detect abrupt temperature changes using outlier detection.

    Args:
        hourly_df: Hourly temperature DataFrame.

    Returns:
        Tuple of (abrupt_changes DataFrame, outlier_threshold).
    """
    hourly_df['hourly_temp_diff'] = hourly_df['temperature'].diff().abs()
    mean_hourly_diff = hourly_df['hourly_temp_diff'].mean()
    std_hourly_diff = hourly_df['hourly_temp_diff'].std()

    outlier_threshold = (
        mean_hourly_diff + OUTLIER_SIGMA_MULTIPLIER * std_hourly_diff
        if not pd.isna(std_hourly_diff)
        else float('inf')
    )

    abrupt_changes = hourly_df[hourly_df['hourly_temp_diff'] > outlier_threshold]

    return abrupt_changes, outlier_threshold


# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

def print_analysis_header(start_timestamp: datetime, end_timestamp: datetime) -> None:
    """Print the analysis header section.

    Args:
        start_timestamp: Analysis start timestamp.
        end_timestamp: Analysis end timestamp.
    """
    print("\n")
    print(SECTION_SEPARATOR)
    print("TEMPERATURE DATA ANALYSIS")
    print(SECTION_SEPARATOR)
    print()
    print("Analysis Period")
    print(SUBSECTION_SEPARATOR)
    print()
    print(f"Start: {start_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"End:   {end_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print()


def print_basic_statistics(stats: dict) -> None:
    """Print basic temperature statistics.

    Args:
        stats: Dictionary of basic statistics.
    """
    print("\nBasic Temperature Statistics")
    print(SUBSECTION_SEPARATOR)
    print()

    basic_stats_data = {
        'Metric': [
            'Mean Temperature',
            'Median Temperature',
            'Max Temperature',
            'Min Temperature',
            'Standard Deviation'
        ],
        'Value (°F)': [
            stats['mean'],
            stats['median'],
            stats['max'],
            stats['min'],
            stats['std_dev']
        ]
    }
    basic_stats_df = pd.DataFrame(basic_stats_data).set_index('Metric')
    basic_stats_df['Value (°F)'] = basic_stats_df['Value (°F)'].map('{:.2f}'.format)
    print(basic_stats_df.to_string())
    print()


def print_trend_analysis(
    rising: TrendMetrics,
    cooling: TrendMetrics,
    heating_dd: float,
    cooling_dd: float,
    baseline_temp: float
) -> None:
    """Print trend analysis summary.

    Args:
        rising: Rising trend metrics.
        cooling: Cooling trend metrics.
        heating_dd: Heating degree days.
        cooling_dd: Cooling degree days.
        baseline_temp: Baseline temperature for degree days.
    """
    print("\nTrend Analysis Summary")
    print(SECTION_SEPARATOR)
    print()

    trend_summary_data = {
        'Trend Type': ['Rising Trends', 'Cooling Trends'],
        'Avg Rate (°F/hour)': [rising.avg_rate, cooling.avg_rate],
        'Avg Duration (hours)': [rising.avg_duration, cooling.avg_duration],
        'Avg Range (°F)': [rising.avg_range, cooling.avg_range],
        'Avg R-squared': [rising.avg_r_squared, cooling.avg_r_squared]
    }
    trend_summary_df = pd.DataFrame(trend_summary_data).set_index('Trend Type')
    trend_summary_df['Avg Rate (°F/hour)'] = (
        trend_summary_df['Avg Rate (°F/hour)'].map('{:.2f}'.format)
    )
    trend_summary_df['Avg Duration (hours)'] = (
        trend_summary_df['Avg Duration (hours)'].map('{:.2f}'.format)
    )
    trend_summary_df['Avg Range (°F)'] = (
        trend_summary_df['Avg Range (°F)'].map('{:.2f}'.format)
    )
    trend_summary_df['Avg R-squared'] = (
        trend_summary_df['Avg R-squared'].map('{:.4f}'.format)
    )
    print(trend_summary_df.to_string())
    print(f"\nHeating Degree Days (Baseline {baseline_temp}°F): {heating_dd:.2f}")
    print(f"Cooling Degree Days (Baseline {baseline_temp}°F): {cooling_dd:.2f}")
    print()


def print_temporal_patterns(patterns: dict) -> None:
    """Print temporal pattern analysis.

    Args:
        patterns: Dictionary of temporal pattern results.
    """
    print("\nTemporal Patterns")
    print(SECTION_SEPARATOR)
    print()
    print("Average Temperature by Hour of Day")
    print(patterns['avg_temp_by_hour'].rename("Avg. Temp (°F)")
          .to_frame().map('{:.2f}'.format).to_string())

    print("\nAverage Temperature by Day of Week")
    print(SUBSECTION_SEPARATOR)
    print()
    print(patterns['avg_temp_by_day_of_week'].rename("Avg. Temp (°F)")
          .to_frame().map('{:.2f}'.format).to_string())

    print(f"\nAverage Daily Temperature Range: {patterns['avg_daily_dtr']:.2f}°F")
    print(f"Maximum Daily Temperature Range: {patterns['max_daily_dtr']:.2f}°F")
    print()


def print_extreme_events(
    heat_spells: SpellMetrics,
    cold_spells: SpellMetrics,
    abrupt_changes: pd.DataFrame,
    outlier_threshold: float
) -> None:
    """Print extreme events and anomaly detection.

    Args:
        heat_spells: Heat spell metrics.
        cold_spells: Cold spell metrics.
        abrupt_changes: DataFrame of abrupt temperature changes.
        outlier_threshold: Threshold used for outlier detection.
    """
    print("\nExtreme Events & Anomaly Detection")
    print(SECTION_SEPARATOR)
    print()

    # Heat spells
    print(f"Heat Spells (Temperature > {HEAT_THRESHOLD_F}°F)")
    print(SUBSECTION_SEPARATOR)
    print()

    heat_spell_data = {
        'Metric': [
            'Number of Heat Spells',
            'Average Duration',
            'Average Intensity (above threshold)'
        ],
        'Value': [
            heat_spells.num_spells,
            heat_spells.avg_duration,
            heat_spells.avg_intensity
        ]
    }
    heat_spell_df = pd.DataFrame(heat_spell_data).set_index('Metric')
    heat_spell_df['Value'] = heat_spell_df.index.map(
        lambda idx: format_spell_metric(idx, heat_spell_df.loc[idx, 'Value'])
    )
    print(heat_spell_df.to_string())

    # Cold spells
    print(f"\nCold Spells (Temperature < {COLD_THRESHOLD_F}°F)")
    print(SUBSECTION_SEPARATOR)
    print()

    cold_spell_data = {
        'Metric': [
            'Number of Cold Spells',
            'Average Duration',
            'Average Intensity (below threshold)'
        ],
        'Value': [
            cold_spells.num_spells,
            cold_spells.avg_duration,
            cold_spells.avg_intensity
        ]
    }
    cold_spell_df = pd.DataFrame(cold_spell_data).set_index('Metric')
    cold_spell_df['Value'] = cold_spell_df.index.map(
        lambda idx: format_spell_metric(idx, cold_spell_df.loc[idx, 'Value'])
    )
    print(cold_spell_df.to_string())

    # Abrupt changes
    print(f"\nAbrupt Temperature Changes (hourly diff > {outlier_threshold:.2f}°F)")
    print(SUBSECTION_SEPARATOR)
    print()
    print(f"Number of Abrupt Changes Detected: {len(abrupt_changes)}")

    if len(abrupt_changes) > 0:
        abrupt_changes_df = abrupt_changes.reset_index()[
            ['timestamp', 'hourly_temp_diff']
        ].copy()
        abrupt_changes_df.columns = ['Timestamp', 'Change (°F)']
        abrupt_changes_df['Timestamp'] = (
            abrupt_changes_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S %Z')
        )
        abrupt_changes_df['Change (°F)'] = (
            abrupt_changes_df['Change (°F)'].map('{:.2f}'.format)
        )
        print(abrupt_changes_df.to_string(index=False))
    else:
        print("No significant abrupt changes detected.")

    print("\n")
    print("--- ANALYSIS COMPLETE ---")
    print()


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def calculate_temperature_statistics(
    df: pd.DataFrame,
    baseline_temp: float = BASELINE_TEMP_F
) -> None:
    """Calculate and print various statistics and analyses for temperature data.

    Args:
        df: DataFrame containing 'entity_id', 'state', and 'last_changed' columns.
        baseline_temp: The baseline temperature in °F for Degree Days.
    """
    # Determine local timezone
    local_tz = datetime.now().astimezone().tzinfo

    # Preprocess data
    df = preprocess_temperature_data(df, local_tz)
    if df is None:
        return

    start_timestamp = df['timestamp'].iloc[0]
    end_timestamp = df['timestamp'].iloc[-1]

    # Calculate basic statistics
    basic_stats = calculate_basic_statistics(df)

    # Add moving average
    df['moving_avg'] = df['temperature'].rolling(
        window=MOVING_AVG_WINDOW, min_periods=1
    ).mean()

    # Analyze trends
    rising_metrics, cooling_metrics = analyze_trends(df)

    # Prepare hourly data
    hourly_df = prepare_hourly_data(df)
    if hourly_df is None:
        # Print what we have and exit
        print_analysis_header(start_timestamp, end_timestamp)
        print_basic_statistics(basic_stats)
        return

    # Calculate degree days
    heating_dd, cooling_dd = calculate_degree_days(hourly_df, baseline_temp)

    # Analyze temporal patterns
    temporal_patterns = analyze_temporal_patterns(hourly_df)

    # Analyze extreme events
    heat_spells = calculate_spell_metrics(
        hourly_df.copy(), HEAT_THRESHOLD_F, is_above=True
    )
    cold_spells = calculate_spell_metrics(
        hourly_df.copy(), COLD_THRESHOLD_F, is_above=False
    )
    abrupt_changes, outlier_threshold = detect_abrupt_changes(hourly_df)

    # Print all results
    print_analysis_header(start_timestamp, end_timestamp)
    print_basic_statistics(basic_stats)
    print_trend_analysis(
        rising_metrics, cooling_metrics, heating_dd, cooling_dd, baseline_temp
    )
    print_temporal_patterns(temporal_patterns)
    print_extreme_events(heat_spells, cold_spells, abrupt_changes, outlier_threshold)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main() -> None:
    """Main entry point for the temperature analysis script."""
    parser = argparse.ArgumentParser(
        description="Analyzes temperature data from a CSV file OR directly from "
                    "Home Assistant.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Create mutually exclusive group for CSV vs. Home Assistant input
    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "--csv-file",
        help="Path to the CSV file containing 'entity_id', 'state', and "
             "'last_changed' columns."
    )
    input_group.add_argument(
        "--ha-sensor",
        help="Entity ID of the Home Assistant temperature sensor "
             "(e.g., 'sensor.home_temperature')."
    )

    parser.add_argument(
        "--ha-url",
        help="Base URL of your Home Assistant instance "
             "(e.g., 'http://homeassistant.local:8123'). Required with --ha-sensor."
    )
    parser.add_argument(
        "--ha-token",
        help="Home Assistant Long-Lived Access Token.\n"
             "Alternatively, set the HA_TOKEN environment variable."
    )
    parser.add_argument(
        "--start-date",
        type=parse_start_date,
        default=(datetime.now() - timedelta(days=DEFAULT_LOOKBACK_DAYS)).replace(
            hour=0, minute=0, second=0, microsecond=0
        ),
        help=f"Start date for Home Assistant data pull (YYYY-MM-DD). "
             f"Defaults to {DEFAULT_LOOKBACK_DAYS} days ago."
    )
    parser.add_argument(
        "--end-date",
        type=parse_end_date,
        default=datetime.now().replace(
            hour=23, minute=59, second=59, microsecond=999999
        ),
        help="End date for Home Assistant data pull (YYYY-MM-DD). Defaults to today."
    )

    args = parser.parse_args()

    try:
        if args.csv_file:
            print(f"Loading data from CSV: {args.csv_file}")
            data_df = pd.read_csv(args.csv_file)
        elif args.ha_sensor:
            if not args.ha_url:
                raise ValueError("When using --ha-sensor, --ha-url is required.")

            ha_token_val = args.ha_token or os.environ.get('HA_TOKEN')
            if not ha_token_val:
                print("\nHome Assistant Long-Lived Access Token not provided.")
                print("You can generate one in HA under Profile -> "
                      "Long-Lived Access Tokens.")
                ha_token_val = input("Please enter your HA Token: ").strip()
                if not ha_token_val:
                    raise ValueError("Home Assistant token is required to fetch data.")

            data_df = fetch_data_from_home_assistant(
                ha_url=args.ha_url,
                ha_token=ha_token_val,
                sensor_name=args.ha_sensor,
                start_date=args.start_date,
                end_date=args.end_date
            )
            if data_df.empty:
                sys.exit(1)
        else:
            # This shouldn't happen due to required=True, but just in case
            raise ValueError("Either --csv-file or --ha-sensor must be provided.")

        calculate_temperature_statistics(df=data_df)

    except FileNotFoundError:
        print(f"Error: The file '{args.csv_file}' was not found.")
        sys.exit(1)
    except ValueError as ve:
        print(f"Configuration Error: {ve}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
