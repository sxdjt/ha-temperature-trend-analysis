import pandas as pd
from scipy import stats
import sys

def calculate_temperature_statistics(file_path):
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

    # Calculate time difference in hours
    df['hours'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / 3600

    # Basic statistics
    mean_temp = df['temperature'].mean()
    median_temp = df['temperature'].median()
    std_dev_temp = df['temperature'].std()
    max_temp = df['temperature'].max()
    min_temp = df['temperature'].min()

    # Identify temperature trends (rising or cooling)
    df['temp_diff'] = df['temperature'].diff()
    df['trend'] = df['temp_diff'].apply(lambda x: 'rising' if x > 0 else ('cooling' if x < 0 else None))

    # Drop the first row (NaN for `temp_diff`)
    df = df.dropna()

    # Group by trends
    trend_groups = df.groupby((df['trend'] != df['trend'].shift()).cumsum())

    # Initialize results
    rise_rate, cool_rate = [], []
    rise_durations, cool_durations = [], []
    rise_ranges, cool_ranges = [], []
    rise_r_squared, cool_r_squared = [], []

    # Process each trend segment
    for _, group in trend_groups:
        total_temp_change = group['temperature'].iloc[-1] - group['temperature'].iloc[0]
        total_hours = group['hours'].iloc[-1] - group['hours'].iloc[0]

        # Skip segments with zero duration
        if total_hours == 0:
            continue

        temp_range = group['temperature'].max() - group['temperature'].min()

        if group['trend'].iloc[0] == 'rising':
            rise_rate.append(total_temp_change / total_hours)
            rise_durations.append(total_hours)
            rise_ranges.append(temp_range)

            # Calculate R-squared for rising trend
            if len(group) > 1:  # R-squared requires at least 2 data points
                slope, intercept, r_value, _, _ = stats.linregress(group['hours'], group['temperature'])
                rise_r_squared.append(r_value ** 2)

        elif group['trend'].iloc[0] == 'cooling':
            cool_rate.append(total_temp_change / total_hours)
            cool_durations.append(total_hours)
            cool_ranges.append(temp_range)

            # Calculate R-squared for cooling trend
            if len(group) > 1:  # R-squared requires at least 2 data points
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

    # Add a moving average (e.g., 1-hour window)
    df['moving_avg'] = df['temperature'].rolling(window=3, min_periods=1).mean()

    # Output results
    print(f"Mean Temperature: {mean_temp:.2f}°F")
    print(f"Median Temperature: {median_temp:.2f}°F")
    print(f"Max Temperature: {max_temp:.2f}°F")
    print(f"Min Temperature: {min_temp:.2f}°F\n")
    
    print(f"Average rate of rise: {avg_rise_rate:.2f}°F/hour")
    print(f"Average rate of cooling: {avg_cool_rate:.2f}°F/hour")
    print(f"Average duration of rising trends: {avg_rise_duration:.2f} hours")
    print(f"Average duration of cooling trends: {avg_cool_duration:.2f} hours")
    print(f"Average temperature range during rise: {avg_rise_range:.2f}°F")
    print(f"Average temperature range during cooling: {avg_cool_range:.2f}°F\n")

    print(f"R-squared for rising trends: {avg_rise_r_squared:.4f}")
    print(f"R-squared for cooling trends: {avg_cool_r_squared:.4f}")

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
