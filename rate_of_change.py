import pandas as pd
from scipy import stats
import sys
import numpy as np

def calculate_temperature_statistics(file_path, baseline_temp=65):
    # Read the data from CSV
    df = pd.read_csv(file_path)

    # Ensure columns are in the expected format
    if not all(col in df.columns for col in ['entity_id', 'state', 'last_changed']):
        raise ValueError("The CSV must contain 'entity_id', 'state', and 'last_changed' columns.")

    # Filter out rows with 'unavailable', 'unknown', or any non-numeric states
    df = df[~df['state'].isin(['unavailable', 'unknown'])]
    df['state'] = pd.to_numeric(df['state'], errors='coerce')
    df = df.dropna(subset=['state'])  # Remove rows where conversion failed

    # Convert types
    df['temperature'] = df['state'].astype(float)
    df['timestamp'] = pd.to_datetime(df['last_changed'])

    # Sort by timestamp to ensure correct time differences
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # Calculate time difference in hours for trend analysis
    df['hours'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / 3600

    # Basic statistics
    mean_temp = df['temperature'].mean()
    median_temp = df['temperature'].median()
    std_dev_temp = df['temperature'].std()
    max_temp = df['temperature'].max()
    min_temp = df['temperature'].min()

    # --- Trend Analysis (using original high-frequency data) ---
    df['temp_diff'] = df['temperature'].diff()
    df['trend'] = df['temp_diff'].apply(lambda x: 'rising' if x > 0 else ('cooling' if x < 0 else None))
    df_trends = df.dropna(subset=['trend']).copy() # Drop first row (NaN for temp_diff) and rows where trend is None (no change)

    # Group by trends
    trend_groups = df_trends.groupby((df_trends['trend'] != df_trends['trend'].shift()).cumsum())

    # Initialize results for trends
    rise_rate, cool_rate = [], []
    rise_durations, cool_durations = [], []
    rise_ranges, cool_ranges = [], []
    rise_r_squared, cool_r_squared = [], []

    # Process each trend segment
    for _, group in trend_groups:
        if len(group) < 2: # Need at least two points to define a trend
            continue

        total_temp_change = group['temperature'].iloc[-1] - group['temperature'].iloc[0]
        total_hours = group['hours'].iloc[-1] - group['hours'].iloc[0]

        # Skip segments with zero duration or zero temp change
        if total_hours == 0 or total_temp_change == 0:
            continue

        temp_range = group['temperature'].max() - group['temperature'].min()

        if group['trend'].iloc[0] == 'rising':
            rise_rate.append(total_temp_change / total_hours)
            rise_durations.append(total_hours)
            rise_ranges.append(temp_range)

            # Calculate R-squared for rising trend
            slope, intercept, r_value, _, _ = stats.linregress(group['hours'], group['temperature'])
            rise_r_squared.append(r_value ** 2)

        elif group['trend'].iloc[0] == 'cooling':
            cool_rate.append(total_temp_change / total_hours)
            cool_durations.append(total_hours)
            cool_ranges.append(temp_range)

            # Calculate R-squared for cooling trend
            slope, intercept, r_value, _, _ = stats.linregress(group['hours'], group['temperature'])
            cool_r_squared.append(r_value ** 2)

    # Calculate averages for both trends
    avg_rise_rate = sum(rise_rate) / len(rise_rate) if rise_rate else 0
    avg_cool_rate = sum(cool_rate) / len(cool_rate) if cool_rate else 0
    avg_rise_duration = sum(rise_durations) / len(rise_durations) if rise_durations else 0
    avg_cool_duration = sum(cool_durations) / len(cool_durations) if cool_durations else 0
    avg_rise_range = sum(rise_ranges) / len(rise_ranges) if rise_ranges else 0
    avg_cool_range = sum(cool_ranges) / len(cool_ranges) if cool_ranges else 0
    avg_rise_r_squared = sum(rise_r_squared) / len(rise_r_squared) if rise_r_squared else 0
    avg_cool_r_squared = sum(cool_r_squared) / len(cool_r_squared) if cool_r_squared else 0

    # Add a moving average (e.g., 3-point window)
    df['moving_avg'] = df['temperature'].rolling(window=3, min_periods=1).mean()

    # --- Prepare data for daily/hourly analysis by resampling to hourly means ---
    # Set timestamp as index and resample to hourly means
    hourly_df = df.set_index('timestamp')['temperature'].resample('H').mean().to_frame()
    hourly_df.columns = ['temperature']
    hourly_df = hourly_df.dropna() # Drop hours with no data

    if hourly_df.empty:
        print("Not enough hourly data to perform detailed daily/hourly analysis.")
        return

    # Calculate daily average temperature for Degree Days and other daily metrics
    hourly_df['date'] = hourly_df.index.date
    daily_avg_temp = hourly_df.groupby('date')['temperature'].mean()

    # Calculate Heating Degree Days (HDD)
    heating_degree_days = (baseline_temp - daily_avg_temp).apply(lambda x: max(0, x)).sum()

    # Calculate Cooling Degree Days (CDD)
    cooling_degree_days = (daily_avg_temp - baseline_temp).apply(lambda x: max(0, x)).sum()

    # --- Temporal Patterns & Cycles (using resampled hourly data) ---
    hourly_df['day_of_week'] = hourly_df.index.day_name()
    hourly_df['hour_of_day'] = hourly_df.index.hour
    hourly_df['month'] = hourly_df.index.month_name()

    avg_temp_by_hour = hourly_df.groupby('hour_of_day')['temperature'].mean()
    avg_temp_by_day_of_week = hourly_df.groupby('day_of_week')['temperature'].mean()
    avg_temp_by_month = hourly_df.groupby('month')['temperature'].mean()

    # Diurnal Temperature Range (DTR)
    daily_max_temp = hourly_df.groupby('date')['temperature'].max()
    daily_min_temp = hourly_df.groupby('date')['temperature'].min()
    daily_dtr = daily_max_temp - daily_min_temp
    avg_daily_dtr = daily_dtr.mean() if not daily_dtr.empty else 0
    max_daily_dtr = daily_dtr.max() if not daily_dtr.empty else 0

    # Lagged Correlations (Autocorrelation)
    # Check if there's enough data for meaningful autocorrelation
    if len(hourly_df) > 1:
        lag_1_correlation = hourly_df['temperature'].autocorr(lag=1) # 1-hour lag
        # You can add more lags if desired, e.g., lag=24 for daily cycle
        lag_24_correlation = hourly_df['temperature'].autocorr(lag=24) if len(hourly_df) > 24 else None
    else:
        lag_1_correlation = None
        lag_24_correlation = None

    # --- Extreme Events & Anomaly Detection (using resampled hourly data) ---

    # Heat/Cold Spells (Example thresholds: Heat > 80F, Cold < 40F)
    heat_threshold = 80
    cold_threshold = 50

    # Heat Spells
    hourly_df['is_hot'] = hourly_df['temperature'] > heat_threshold
    # Identify contiguous blocks of 'True' (hot)
    hourly_df['heat_spell_id'] = (hourly_df['is_hot'] != hourly_df['is_hot'].shift()).cumsum()
    heat_spells = hourly_df[hourly_df['is_hot']].groupby('heat_spell_id')

    num_heat_spells = len(heat_spells)
    avg_heat_spell_duration = 0
    avg_heat_spell_intensity = 0

    if num_heat_spells > 0:
        heat_spell_durations = [len(group) for _, group in heat_spells]
        heat_spell_intensities = [group['temperature'].mean() - heat_threshold for _, group in heat_spells]
        avg_heat_spell_duration = sum(heat_spell_durations) / num_heat_spells
        avg_heat_spell_intensity = sum(heat_spell_intensities) / num_heat_spells

    # Cold Spells
    hourly_df['is_cold'] = hourly_df['temperature'] < cold_threshold
    # Identify contiguous blocks of 'True' (cold)
    hourly_df['cold_spell_id'] = (hourly_df['is_cold'] != hourly_df['is_cold'].shift()).cumsum()
    cold_spells = hourly_df[hourly_df['is_cold']].groupby('cold_spell_id')

    num_cold_spells = len(cold_spells)
    avg_cold_spell_duration = 0
    avg_cold_spell_intensity = 0

    if num_cold_spells > 0:
        cold_spell_durations = [len(group) for _, group in cold_spells]
        cold_spell_intensities = [cold_threshold - group['temperature'].mean() for _, group in cold_spells]
        avg_cold_spell_duration = sum(cold_spell_durations) / num_cold_spells
        avg_cold_spell_intensity = sum(cold_spell_intensities) / num_cold_spells


    # Abrupt Changes/Outliers (using 3-sigma rule on hourly differences)
    hourly_df['hourly_temp_diff'] = hourly_df['temperature'].diff().abs()
    mean_hourly_diff = hourly_df['hourly_temp_diff'].mean()
    std_hourly_diff = hourly_df['hourly_temp_diff'].std()
    # Define an outlier as a change more than 3 standard deviations from the mean change
    outlier_threshold = mean_hourly_diff + 3 * std_hourly_diff if not np.isnan(std_hourly_diff) else float('inf')

    abrupt_changes = hourly_df[hourly_df['hourly_temp_diff'] > outlier_threshold]
    num_abrupt_changes = len(abrupt_changes)

    # --- Output Results ---
    print("--- Overall Temperature Statistics ---")
    print(f"Mean Temperature: {mean_temp:.2f}°F")
    print(f"Median Temperature: {median_temp:.2f}°F")
    print(f"Max Temperature: {max_temp:.2f}°F")
    print(f"Min Temperature: {min_temp:.2f}°F")
    print(f"Standard Deviation: {std_dev_temp:.2f}°F\n")

    print("--- Trend Analysis ---")
    print(f"Average rate of rise: {avg_rise_rate:.2f}°F/hour")
    print(f"Average rate of cooling: {avg_cool_rate:.2f}°F/hour")
    print(f"Average duration of rising trends: {avg_rise_duration:.2f} hours")
    print(f"Average duration of cooling trends: {avg_cool_duration:.2f} hours")
    print(f"Average temperature range during rise: {avg_rise_range:.2f}°F")
    print(f"Average temperature range during cooling: {avg_cool_range:.2f}°F")
    print(f"R-squared for rising trends: {avg_rise_r_squared:.4f}")
    print(f"R-squared for cooling trends: {avg_cool_r_squared:.4f}\n")

    print(f"Total Heating Degree Days (Baseline {baseline_temp}°F): {heating_degree_days:.2f}")
    print(f"Total Cooling Degree Days (Baseline {baseline_temp}°F): {cooling_degree_days:.2f}\n")

    print("--- Temporal Patterns ---")
    print("Average Temperature by Hour of Day:")
    print(avg_temp_by_hour.map('{:.2f}°F'.format).to_string())
    print("\nAverage Temperature by Day of Week:")
    print(avg_temp_by_day_of_week.map('{:.2f}°F'.format).to_string())
    print("\nAverage Temperature by Month:")
    print(avg_temp_by_month.map('{:.2f}°F'.format).to_string())
    print(f"\nAverage Daily Temperature Range (DTR): {avg_daily_dtr:.2f}°F")
    print(f"Maximum Daily Temperature Range (DTR): {max_daily_dtr:.2f}°F")

    print("\n--- Time Series Correlation ---")
    if lag_1_correlation is not None:
        print(f"Autocorrelation with 1-hour lag: {lag_1_correlation:.4f}")
    else:
        print("Not enough data for 1-hour lag autocorrelation.")

    if lag_24_correlation is not None:
        print(f"Autocorrelation with 24-hour lag (daily cycle): {lag_24_correlation:.4f}")
    else:
        print("Not enough data for 24-hour lag autocorrelation.")


    print("\n--- Extreme Events & Anomaly Detection ---")
    print(f"Heat Spells (Temperature > {heat_threshold}°F):")
    print(f"  Number of Heat Spells: {num_heat_spells}")
    print(f"  Average Heat Spell Duration: {avg_heat_spell_duration:.2f} hours")
    print(f"  Average Heat Spell Intensity (above threshold): {avg_heat_spell_intensity:.2f}°F")

    print(f"\nCold Spells (Temperature < {cold_threshold}°F):")
    print(f"  Number of Cold Spells: {num_cold_spells}")
    print(f"  Average Cold Spell Duration: {avg_cold_spell_duration:.2f} hours")
    print(f"  Average Cold Spell Intensity (below threshold): {avg_cold_spell_intensity:.2f}°F")

    print(f"\nAbrupt Temperature Changes (hourly diff > {outlier_threshold:.2f}°F):")
    print(f"  Number of Abrupt Changes Detected: {num_abrupt_changes}")
    if num_abrupt_changes > 0:
        print("  Timestamps of Abrupt Changes:")
        for idx, row in abrupt_changes.iterrows():
            print(f"    - {row.name.strftime('%Y-%m-%d %H:%M')}: Change of {row['hourly_temp_diff']:.2f}°F")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <csv_file_path>")
        sys.exit(1)

    csv_file_path = sys.argv[1]
    try:
        calculate_temperature_statistics(csv_file_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)