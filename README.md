A first attempt at creating rate of change and other statistics, using Home Assitant historical data.  Basically, I want to see how fast my garage heats and cools.

_Note: The script defaults to pulling the last 30 days of data; my HA integration was reset recently and only has a few days' worth._

```
python rate_of_change.py --ha-sensor sensor.apollo_air_1_garage_sen55_temperature --ha-url http://homeassistant.local:8123 --ha-token YOUR-HA-TOKEN-HERE

Fetching data from Home Assistant for 'sensor.apollo_air_1_garage_sen55_temperature' from 2025-05-07 07:00 to 2025-06-07 06:59 (UTC)...
Successfully fetched 446 records from Home Assistant.
Filtered out 11 non-numeric/unavailable temperature readings.


============================================================
TEMPERATURE DATA ANALYSIS
============================================================

Analysis Period
------------------------------------------------------------
Start: 2025-06-01 07:39:39 UTC
End:   2025-06-06 08:01:18 UTC


Basic Temperature Statistics
------------------------------------------------------------
                   Value (°F)
Metric                       
Mean Temperature        69.78
Median Temperature      69.80
Max Temperature         75.20
Min Temperature         63.50
Standard Deviation       2.68


Trend Analysis Summary
============================================================

               Avg Rate (°F/hour) Avg Duration (hours) Avg Range (°F) Avg R-squared
Trend Type                                                                         
Rising Trends                1.28                 2.59           2.57        0.9544
Cooling Trends              -0.83                 4.83           2.44        0.9488

Heating Degree Days (Baseline 65°F): 0.00
Cooling Degree Days (Baseline 65°F): 28.28

Temporal Patterns
============================================================

Average Temperature by Hour of Day
            Avg. Temp (°F)
hour_of_day               
0                    71.27
1                    72.33
2                    73.24
3                    72.96
4                    72.16
5                    71.88
6                    71.53
7                    70.49
8                    69.93
9                    68.91
10                   68.86
11                   68.40
12                   67.25
13                   66.79
14                   66.41
15                   66.08
16                   66.20
17                   65.91
18                   65.97
19                   66.27
20                   66.97
21                   68.14
22                   69.08
23                   70.21

Average Temperature by Day of Week
------------------------------------------------------------
          Avg. Temp (°F)
Monday             67.98
Tuesday            69.22
Wednesday          71.82
Thursday           69.47
Friday             73.41
Saturday             nan
Sunday             66.39

Average Daily Temperature Range: 5.30°F
Maximum Daily Temperature Range: 7.56°F


Extreme Events & Anomaly Detection
============================================================

Heat Spells (Temperature > 80°F)
------------------------------------------------------------
                                          Value
Metric                                         
Number of Heat Spells                         0
Average Duration                     0.00 hours
Average Intensity (above threshold)      0.00°F

Cold Spells (Temperature < 40°F)
------------------------------------------------------------
                                          Value
Metric                                         
Number of Cold Spells                         0
Average Duration                     0.00 hours
Average Intensity (below threshold)      0.00°F

Abrupt Temperature Changes (hourly diff > 1.89°F)
------------------------------------------------------------
Number of Abrupt Changes Detected: 1
           Timestamp Change (°F)
2025-06-01 17:00 UTC        2.10


--- ANALYSIS COMPLETE ---

```
