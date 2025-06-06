import pandas as pd
from scipy import stats
import sys
import numpy as np
import argparse
from datetime import datetime, timedelta, timezone
import requests
import os # For environment variables

def fetch_data_from_home_assistant(ha_url: str, ha_token: str, sensor_name: str,
                                   start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetches historical temperature data from Home Assistant.

    Args:
        ha_url (str): The base URL of your Home Assistant instance (e.g., http://homeassistant.local:8123).
        ha_token (str): Your Home Assistant Long-Lived Access Token.
        sensor_name (str): The entity_id of the temperature sensor (e.g., "sensor.home_temperature").
        start_date (datetime): The start datetime for fetching data (will be converted to UTC).
        end_date (datetime): The end datetime for fetching data (will be converted to UTC).

    Returns:
        pd.DataFrame: A DataFrame containing 'entity_id', 'state', and 'last_changed' columns,
                      or an empty DataFrame if no data is found or an error occurs.
    """
    
    # Ensure dates are timezone-aware UTC for HA API
    # If the input datetimes from argparse are naive, assume they are local and convert to UTC
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
        "minimal_response": "true" # Get only state, last_changed, context
    }

    print(f"Fetching data from Home Assistant for '{sensor_name}' from {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')} (UTC)...")

    try:
        response = requests.get(api_url, headers=headers, params=params, timeout=30)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        data = response.json()

        if not data or not isinstance(data, list) or not data[0]:
            print(f"No data found for sensor '{sensor_name}' in the specified period from Home Assistant.")
            return pd.DataFrame()

        # HA returns a list of lists (grouped by entity_id, even if just one)
        # We expect data[0] to be the list of states for our sensor
        sensor_states = data[0]

        extracted_data = []
        for state_obj in sensor_states:
            extracted_data.append({
                'entity_id': sensor_name, # The API might not always return this directly in each state object for minimal_response
                'state': state_obj.get('state'),
                'last_changed': state_obj.get('last_changed')
            })

        df = pd.DataFrame(extracted_data)
        print(f"Successfully fetched {len(df)} records from Home Assistant.")
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Home Assistant: {e}")
        print(f"Please ensure the URL is correct, Home Assistant is running, and your token is valid.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred while fetching HA data: {e}")
        return pd.DataFrame()

def calculate_temperature_statistics(df: pd.DataFrame, baseline_temp: float = 65) -> None:
    """
    Calculates and prints various statistics and analyses for temperature data
    from a DataFrame in plain text format.

    Args:
        df (pd.DataFrame): The DataFrame containing 'entity_id', 'state', 'last_changed' columns.
        baseline_temp (float, optional): The baseline temperature in °F for Degree Days. Defaults to 65°F.

    Raises:
        ValueError: If the DataFrame does not contain the required columns.
        Exception: For other errors during data analysis.
    """
    # Plain text formatting variables
    H1 = "=" * 60 # For top/bottom separators
    H2 = "" # No markdown H2, just plain
    H3 = "" # No markdown H3, just plain
    
    # --- Data Preprocessing ---
    required_columns = ['entity_id', 'state', 'last_changed']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"The input DataFrame must contain {', '.join(required_columns)} columns.")

    initial_rows = len(df)
    # Apply .copy() here to ensure you're working on a new DataFrame
    df = df[~df['state'].isin(['unavailable', 'unknown', 'None', 'null'])].copy()
    df['state'] = pd.to_numeric(df['state'], errors='coerce')
    df = df.dropna(subset=['state'])
    print(f"Filtered out {initial_rows - len(df)} non-numeric/unavailable temperature readings.")

    df['temperature'] = df['state'].astype(float)
    df['timestamp'] = pd.to_datetime(df['last_changed'])

    df = df.sort_values(by='timestamp').reset_index(drop=True)

    if df.empty:
        print("No valid temperature data remaining after cleaning. Exiting analysis.")
        return

    start_timestamp = df['timestamp'].iloc[0]
    end_timestamp = df['timestamp'].iloc[-1]

    df['hours'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / 3600

    # --- Basic Statistics ---
    mean_temp = df['temperature'].mean()
    median_temp = df['temperature'].median()
    std_dev_temp = df['temperature'].std()
    max_temp = df['temperature'].max()
    min_temp = df['temperature'].min()

    df['moving_avg'] = df['temperature'].rolling(window=3, min_periods=1).mean()

    # --- Trend Analysis ---
    df['temp_diff'] = df['temperature'].diff()
    df['trend'] = df['temp_diff'].apply(lambda x: 'rising' if x > 0 else ('cooling' if x < 0 else None))
    df_trends = df.dropna(subset=['trend']).copy()

    trend_groups = df_trends.groupby((df_trends['trend'] != df_trends['trend'].shift()).cumsum())

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

        slope, intercept, r_value, p_value, std_err = stats.linregress(group['hours'], group['temperature'])
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

    avg_rise_rate = sum(rise_rate) / len(rise_rate) if rise_rate else 0
    avg_cool_rate = sum(cool_rate) / len(cool_rate) if cool_rate else 0
    avg_rise_duration = sum(rise_durations) / len(rise_durations) if rise_durations else 0
    avg_cool_duration = sum(cool_durations) / len(cool_durations) if cool_durations else 0
    avg_rise_range = sum(rise_ranges) / len(rise_ranges) if rise_ranges else 0
    avg_cool_range = sum(cool_ranges) / len(cool_ranges) if cool_ranges else 0
    avg_rise_r_squared = sum(rise_r_squared) / len(rise_r_squared) if rise_r_squared else 0
    avg_cool_r_squared = sum(cool_r_squared) / len(cool_r_squared) if cool_r_squared else 0

    # --- Prepare data for daily/hourly analysis by resampling to hourly means ---
    # Ensure timezone-aware resampling
    # Convert 'timestamp' to a timezone-aware index for resampling
    # If timestamps are naive, assume they are in local system timezone for resampling
    # and then convert to UTC for consistent display in the report.
    if df['timestamp'].dt.tz is None:
        # If naive, localize to current system timezone (useful for HA data which might be local)
        try:
            local_tz = datetime.now().astimezone().tzinfo
            df['timestamp'] = df['timestamp'].dt.tz_localize(local_tz)
        except Exception:
            # Fallback if tz_localize fails, use UTC
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            print("Warning: Could not localize timestamp to system timezone; assuming UTC for resampling.")
    
    hourly_df = df.set_index('timestamp')['temperature'].resample('h').mean().to_frame()
    hourly_df.columns = ['temperature']
    hourly_df = hourly_df.dropna()
    
    # Convert back to UTC for consistent display in report, if not already
    if hourly_df.index.tz is not None and hourly_df.index.tz != timezone.utc:
        hourly_df = hourly_df.tz_convert('UTC')
    elif hourly_df.index.tz is None: # If it's still naive after resampling
         hourly_df = hourly_df.tz_localize('UTC')


    if hourly_df.empty:
        print("Not enough hourly data to perform detailed daily/hourly analysis. Some sections will be skipped.")
        pass

    hourly_df['date'] = hourly_df.index.date
    daily_avg_temp = hourly_df.groupby('date')['temperature'].mean()

    heating_degree_days = (baseline_temp - daily_avg_temp).apply(lambda x: max(0, x)).sum()
    cooling_degree_days = (daily_avg_temp - baseline_temp).apply(lambda x: max(0, x)).sum()

    # --- Temporal Patterns & Cycles ---
    hourly_df['day_of_week'] = hourly_df.index.day_name()
    hourly_df['hour_of_day'] = hourly_df.index.hour

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
    avg_daily_dtr = daily_dtr.mean() if not daily_dtr.empty else 0
    max_daily_dtr = daily_dtr.max() if not daily_dtr.empty else 0

    # Lagged Correlations (Autocorrelation)
    lag_1_correlation = hourly_df['temperature'].autocorr(lag=1) if len(hourly_df) > 1 else None
    lag_24_correlation = hourly_df['temperature'].autocorr(lag=24) if len(hourly_df) > 24 else None

    # --- Extreme Events & Anomaly Detection ---
    heat_threshold = 80
    cold_threshold = 40

    hourly_df['is_hot'] = hourly_df['temperature'] > heat_threshold
    hourly_df['heat_spell_id'] = (hourly_df['is_hot'] != hourly_df['is_hot'].shift(1)).cumsum()
    heat_spells_data = hourly_df[hourly_df['is_hot']]
    heat_spells_grouped = heat_spells_data.groupby('heat_spell_id')

    num_heat_spells = len(heat_spells_grouped) if not heat_spells_data.empty else 0
    avg_heat_spell_duration = 0
    avg_heat_spell_intensity = 0

    if num_heat_spells > 0:
        heat_spell_durations = [len(group) for _, group in heat_spells_grouped]
        heat_spell_intensities = [group['temperature'].mean() - heat_threshold for _, group in heat_spells_grouped]
        avg_heat_spell_duration = sum(heat_spell_durations) / num_heat_spells
        avg_heat_spell_intensity = sum(heat_spell_intensities) / num_heat_spells

    hourly_df['is_cold'] = hourly_df['temperature'] < cold_threshold
    hourly_df['cold_spell_id'] = (hourly_df['is_cold'] != hourly_df['is_cold'].shift(1)).cumsum()
    cold_spells_data = hourly_df[hourly_df['is_cold']]
    cold_spells_grouped = cold_spells_data.groupby('cold_spell_id')

    num_cold_spells = len(cold_spells_grouped) if not cold_spells_data.empty else 0
    avg_cold_spell_duration = 0
    avg_cold_spell_intensity = 0

    if num_cold_spells > 0:
        cold_spell_durations = [len(group) for _, group in cold_spells_grouped]
        cold_spell_intensities = [cold_threshold - group['temperature'].mean() for _, group in cold_spells_grouped]
        avg_cold_spell_duration = sum(cold_spell_durations) / num_cold_spells
        avg_cold_spell_intensity = sum(cold_spell_intensities) / num_cold_spells

    hourly_df['hourly_temp_diff'] = hourly_df['temperature'].diff().abs()
    mean_hourly_diff = hourly_df['hourly_temp_diff'].mean()
    std_hourly_diff = hourly_df['hourly_temp_diff'].std()

    outlier_threshold = mean_hourly_diff + 3 * std_hourly_diff if not pd.isna(std_hourly_diff) else float('inf')

    abrupt_changes = hourly_df[hourly_df['hourly_temp_diff'] > outlier_threshold]
    num_abrupt_changes = len(abrupt_changes)

    # --- Output Results ---
    print(f"\n")
    print(f"--- TEMPERATURE DATA ANALYSIS REPORT ---")
    print(f"\n")

    print(f"Analysis Period")
    print(f"------------------------------------------------------------")
    print(f"Start: {start_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"End:   {end_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")

    print(f"\nBasic Temperature Statistics")
    print(f"------------------------------------------------------------")
    basic_stats_data = {
        'Metric': ['Mean Temperature', 'Median Temperature', 'Max Temperature', 'Min Temperature', 'Standard Deviation'],
        'Value (°F)': [mean_temp, median_temp, max_temp, min_temp, std_dev_temp]
    }
    basic_stats_df = pd.DataFrame(basic_stats_data).set_index('Metric')
    basic_stats_df['Value (°F)'] = basic_stats_df['Value (°F)'].map('{:.2f}'.format)
    print(basic_stats_df.to_string()) # Changed to to_string()
    print("\n")

    print(f"Trend Analysis Summary")
    print(f"============================================================\n")
    trend_summary_data = {
        'Trend Type': ['Rising Trends', 'Cooling Trends'],
        'Avg Rate (°F/hour)': [avg_rise_rate, avg_cool_rate],
        'Avg Duration (hours)': [avg_rise_duration, avg_cool_duration],
        'Avg Range (°F)': [avg_rise_range, avg_cool_range],
        'Avg R-squared': [avg_rise_r_squared, avg_cool_r_squared]
    }
    trend_summary_df = pd.DataFrame(trend_summary_data).set_index('Trend Type')
    trend_summary_df['Avg Rate (°F/hour)'] = trend_summary_df['Avg Rate (°F/hour)'].map('{:.2f}'.format)
    trend_summary_df['Avg Duration (hours)'] = trend_summary_df['Avg Duration (hours)'].map('{:.2f}'.format)
    trend_summary_df['Avg Range (°F)'] = trend_summary_df['Avg Range (°F)'].map('{:.2f}'.format)
    trend_summary_df['Avg R-squared'] = trend_summary_df['Avg R-squared'].map('{:.4f}'.format)
    print(trend_summary_df.to_string()) # Changed to to_string()

    print(f"\nHeating Degree Days (Baseline {baseline_temp}°F): {heating_degree_days:.2f}")
    print(f"Cooling Degree Days (Baseline {baseline_temp}°F): {cooling_degree_days:.2f}\n")

    print(f"Temporal Patterns")
    print(f"============================================================\n")

    print(f"Average Temperature by Hour of Day")
    print(avg_temp_by_hour.rename("Avg. Temp (°F)").to_frame().map('{:.2f}'.format).to_string()) # Changed to to_string()

    print(f"\nAverage Temperature by Day of Week")
    print(f"------------------------------------------------------------")
    print(avg_temp_by_day_of_week.rename("Avg. Temp (°F)").to_frame().map('{:.2f}'.format).to_string()) # Changed to to_string()

    print(f"\nAverage Daily Temperature Range: {avg_daily_dtr:.2f}°F")
    print(f"Maximum Daily Temperature Range: {max_daily_dtr:.2f}°F\n")

#    print(f"\nTime Series Correlation")
#    print(f"------------------------------------------------------------")
#    if lag_1_correlation is not None and lag_24_correlation is not None:
#        correlation_data = {
#            'Lag': ['1-hour', '24-hour (daily cycle)'],
#            'Autocorrelation': [lag_1_correlation, lag_24_correlation]
#        }
#        correlation_df = pd.DataFrame(correlation_data).set_index('Lag')
#        correlation_df['Autocorrelation'] = correlation_df['Autocorrelation'].map('{:.4f}'.format)
#        print(correlation_df.to_string()) # Changed to to_string()
#    elif lag_1_correlation is not None:
#        print(f"Autocorrelation with 1-hour lag: {lag_1_correlation:.4f}")
#        print(f"Not enough data for 24-hour lag autocorrelation.")
#    else:
#        print(f"Not enough data for meaningful autocorrelation analysis.")
#    print("\n")

    print(f"\nExtreme Events & Anomaly Detection")
    print(f"============================================================\n")
    print(f"Heat Spells (Temperature > {heat_threshold}°F)")
    print(f"------------------------------------------------------------")
    heat_spell_data = {
        'Metric': ['Number of Heat Spells', 'Average Duration', 'Average Intensity (above threshold)'],
        'Value': [num_heat_spells, avg_heat_spell_duration, avg_heat_spell_intensity]
    }
    heat_spell_df = pd.DataFrame(heat_spell_data).set_index('Metric')
    heat_spell_df['Value'] = heat_spell_df['Value'].astype(str)
    # Removed color formatting logic for Value column
    heat_spell_df['Value'] = heat_spell_df.index.map(lambda idx: f"{float(heat_spell_df.loc[idx, 'Value']):.2f} hours" if "Duration" in idx else (f"{float(heat_spell_df.loc[idx, 'Value']):.2f}°F" if "Intensity" in idx else f"{int(float(heat_spell_df.loc[idx, 'Value']))}"))
    print(heat_spell_df.to_string()) # Changed to to_string()


    print(f"\nCold Spells (Temperature < {cold_threshold}°F)")
    print(f"------------------------------------------------------------")
    cold_spell_data = {
        'Metric': ['Number of Cold Spells', 'Average Duration', 'Average Intensity (below threshold)'],
        'Value': [num_cold_spells, avg_cold_spell_duration, avg_cold_spell_intensity]
    }
    cold_spell_df = pd.DataFrame(cold_spell_data).set_index('Metric')
    cold_spell_df['Value'] = cold_spell_df['Value'].astype(str)
    # Removed color formatting logic for Value column
    cold_spell_df['Value'] = cold_spell_df.index.map(lambda idx: f"{float(cold_spell_df.loc[idx, 'Value']):.2f} hours" if "Duration" in idx else (f"{float(cold_spell_df.loc[idx, 'Value']):.2f}°F" if "Intensity" in idx else f"{int(float(cold_spell_df.loc[idx, 'Value']))}"))
    print(cold_spell_df.to_string()) # Changed to to_string()

    print(f"\nAbrupt Temperature Changes (hourly diff > {outlier_threshold:.2f}°F)")
    print(f"------------------------------------------------------------")
    print(f"Number of Abrupt Changes Detected: {num_abrupt_changes}")
    if num_abrupt_changes > 0:
        abrupt_changes_df = abrupt_changes.reset_index()[['timestamp', 'hourly_temp_diff']].copy()
        abrupt_changes_df.columns = ['Timestamp', 'Change (°F)']
        abrupt_changes_df['Timestamp'] = abrupt_changes_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M %Z') # Display timezone
        abrupt_changes_df['Change (°F)'] = abrupt_changes_df['Change (°F)'].map('{:.2f}'.format)
        print(abrupt_changes_df.to_string(index=False)) # Changed to to_string()
    else:
        print(f"No significant abrupt changes detected.")

    print(f"\n")
    print(f"--- ANALYSIS COMPLETE ---")
    print(f"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyzes temperature data from a CSV file OR directly from Home Assistant.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Create a mutually exclusive group for CSV vs. Home Assistant input
    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "--csv-file",
        help="Path to the CSV file containing 'entity_id', 'state', and 'last_changed' columns."
    )
    input_group.add_argument(
        "--ha-sensor",
        help="Entity ID of the Home Assistant temperature sensor (e.g., 'sensor.home_temperature')."
    )

    parser.add_argument(
        "--ha-url",
        help="Base URL of your Home Assistant instance (e.g., 'http://homeassistant.local:8123'). Required with --ha-sensor."
    )
    parser.add_argument(
        "--ha-token",
        help="Home Assistant Long-Lived Access Token. \n"
             "Alternatively, set the HA_TOKEN environment variable."
    )
    parser.add_argument(
        "--start-date",
        type=lambda s: datetime.strptime(s, '%Y-%m-%d').replace(hour=0, minute=0, second=0, microsecond=0), # Start of the day
        default=(datetime.now() - timedelta(days=30)).replace(hour=0, minute=0, second=0, microsecond=0), # Default to 30 days ago, start of day
        help="Start date for Home Assistant data pull (YYYY-MM-DD). Defaults to 30 days ago."
    )
    parser.add_argument(
        "--end-date",
        type=lambda s: datetime.strptime(s, '%Y-%m-%d').replace(hour=23, minute=59, second=59, microsecond=999999), # End of the day
        default=datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999), # Default to today, end of day
        help="End date for Home Assistant data pull (YYYY-MM-DD). Defaults to today."
    )

    # Removed --pretty argument

    args = parser.parse_args()
    # pretty_output is no longer needed

    # --- Data Source Selection and Fetching ---
    data_df = pd.DataFrame()
    try:
        if args.csv_file:
            print(f"Loading data from CSV: {args.csv_file}")
            data_df = pd.read_csv(args.csv_file)
        elif args.ha_sensor:
            if not args.ha_url:
                raise ValueError("When using --ha-sensor, --ha-url is required.")

            ha_token_val = args.ha_token or os.environ.get('HA_TOKEN')
            if not ha_token_val:
                # Prompt if not provided via --ha-token or env var
                print("\nHome Assistant Long-Lived Access Token not provided.")
                print("You can generate one in HA under Profile -> Long-Lived Access Tokens.")
                ha_token_val = input("Please enter your HA Token: ").strip()
                if not ha_token_val:
                    raise ValueError("Home Assistant token is required to fetch data.")

            # Pass datetime objects directly
            data_df = fetch_data_from_home_assistant(
                ha_url=args.ha_url,
                ha_token=ha_token_val,
                sensor_name=args.ha_sensor,
                start_date=args.start_date,
                end_date=args.end_date
                # pretty_output removed
            )
            if data_df.empty:
                sys.exit(1) # Exit if HA data fetching failed or returned no data

        # Now, pass the fetched/loaded DataFrame to the analysis function
        calculate_temperature_statistics(df=data_df) # pretty_output arg removed

    except FileNotFoundError:
        error_msg = f"Error: The file '{args.csv_file}' was not found."
        print(error_msg)
        sys.exit(1)
    except ValueError as ve:
        error_msg = f"Configuration Error: {ve}"
        print(error_msg)
        sys.exit(1)
    except Exception as e:
        error_msg = f"An unexpected error occurred: {e}"
        print(error_msg)
        sys.exit(1)