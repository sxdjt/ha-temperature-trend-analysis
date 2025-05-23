A first attempt at creating rate of change and other statistics, using Home Assitant historical data.  Basically, I want to see how fast my garage heats and cools.


```
python rate_of_change.py history.csv

--- Overall Temperature Statistics ---
Mean Temperature: 63.85°F
Median Temperature: 63.50°F
Max Temperature: 76.70°F
Min Temperature: 59.00°F
Standard Deviation: 2.63°F

--- Trend Analysis ---
Average rate of rise: 1.12°F/hour
Average rate of cooling: -0.67°F/hour
Average duration of rising trends: 2.63 hours
Average duration of cooling trends: 3.41 hours
Average temperature range during rise: 2.42°F
Average temperature range during cooling: 1.75°F
R-squared for rising trends: 0.9616
R-squared for cooling trends: 0.9543

Total Heating Degree Days (Baseline 65°F): 28.98
Total Cooling Degree Days (Baseline 65°F): 18.34

--- Temporal Patterns ---
Average Temperature by Hour of Day:
hour_of_day
0     66.44°F
1     67.04°F
2     67.48°F
3     67.32°F
4     67.01°F
5     66.64°F
6     65.91°F
7     65.30°F
8     64.70°F
9     64.34°F
10    64.04°F
11    63.84°F
12    63.27°F
13    63.14°F
14    62.56°F
15    62.34°F
16    62.34°F
17    62.51°F
18    62.44°F
19    62.91°F
20    63.57°F
21    64.38°F
22    64.91°F
23    65.62°F

Average Temperature by Day of Week:
day_of_week
Friday       65.65°F
Monday       63.20°F
Saturday     65.90°F
Sunday       63.37°F
Thursday     63.88°F
Tuesday      65.17°F
Wednesday    64.82°F

Average Temperature by Month:
month
May    64.59°F

Average Daily Temperature Range (DTR): 6.18°F
Maximum Daily Temperature Range (DTR): 11.22°F

--- Time Series Correlation ---
Autocorrelation with 1-hour lag: 0.9655
Autocorrelation with 24-hour lag (daily cycle): 0.6211

--- Extreme Events & Anomaly Detection ---
Heat Spells (Temperature > 80°F):
  Number of Heat Spells: 0
  Average Heat Spell Duration: 0.00 hours
  Average Heat Spell Intensity (above threshold): 0.00°F

Cold Spells (Temperature < 50°F):
  Number of Cold Spells: 0
  Average Cold Spell Duration: 0.00 hours
  Average Cold Spell Intensity (below threshold): 0.00°F

Abrupt Temperature Changes (hourly diff > 2.32°F):
  Number of Abrupt Changes Detected: 11
  Timestamps of Abrupt Changes:
    - 2025-05-01 11:00: Change of 2.86°F
    - 2025-05-01 23:00: Change of 3.92°F
    - 2025-05-02 00:00: Change of 3.56°F
    - 2025-05-02 01:00: Change of 3.43°F
    - 2025-05-02 12:00: Change of 2.49°F
    - 2025-05-02 19:00: Change of 2.47°F
    - 2025-05-03 06:00: Change of 2.66°F
    - 2025-05-03 21:00: Change of 2.39°F
    - 2025-05-05 21:00: Change of 3.40°F
    - 2025-05-06 14:00: Change of 2.72°F
    - 2025-05-06 21:00: Change of 2.50°F
```
