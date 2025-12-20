# Temperature Trend Analysis

[![AI Assisted](https://img.shields.io/badge/AI-Claude%20Code-AAAAAA.svg)](https://claude.ai/code)

The script uses data from Home Assistant.  It defaults to the last 30 days, but you can specify a date range with ```--start-date``` and ```--end-date``` parameters.

# Requirements

* argparse
* datetime
* numpy
* os 
* pandas
* requests
* scipy
* sys

```
python ha-temp-trend.py --ha-sensor sensor.patio_climate_temperature \
  --ha-url http://homeassistant.local:8123 \
  --ha-token [your long-lived API token]

* Fetching data from Home Assistant for 'sensor.patio_climate_temperature' from 2025-05-26 07:00 to 2025-06-26 06:59 (UTC)...
* Successfully fetched 5470 records from Home Assistant.
* Filtered out 1 non-numeric/unavailable temperature readings.


============================================================
TEMPERATURE DATA ANALYSIS
============================================================

Analysis Period
------------------------------------------------------------

Start: 2025-06-03 04:23:39 PDT
End:   2025-06-25 20:52:21 PDT


Basic Temperature Statistics
------------------------------------------------------------

                   Value (°F)
Metric                       
Mean Temperature        74.61
Median Temperature      73.76
Max Temperature        109.76
Min Temperature         44.06
Standard Deviation      14.66


Trend Analysis Summary
============================================================

               Avg Rate (°F/hour) Avg Duration (hours) Avg Range (°F) Avg R-squared
Trend Type                                                                         
Rising Trends                4.97                 3.24          17.74        0.9471
Cooling Trends              -4.08                 5.29          17.33        0.9641

Heating Degree Days (Baseline 65°F): 23.98
Cooling Degree Days (Baseline 65°F): 135.95

Temporal Patterns
============================================================

Average Temperature by Hour of Day
            Avg. Temp (°F)
hour_of_day               
0                    57.83
1                    56.01
2                    54.46
3                    53.29
4                    52.27
5                    51.41
6                    51.78
7                    56.20
8                    64.52
9                    71.53
10                   77.49
11                   83.08
12                   86.84
13                   88.86
14                   89.12
15                   88.79
16                   86.93
17                   83.80
18                   80.07
19                   75.79
20                   71.22
21                   66.51
22                   62.74
23                   59.98

Average Temperature by Day of Week
------------------------------------------------------------

          Avg. Temp (°F)
Monday             74.52
Tuesday            74.46
Wednesday          68.20
Thursday           69.28
Friday             65.20
Saturday           65.32
Sunday             71.10

Average Daily Temperature Range: 40.47°F
Maximum Daily Temperature Range: 59.04°F


Extreme Events & Anomaly Detection
============================================================

Heat Spells (Temperature > 80°F)
------------------------------------------------------------

                                          Value
Metric                                         
Number of Heat Spells                        19
Average Duration                     8.05 hours
Average Intensity (above threshold)     10.40°F

Cold Spells (Temperature < 40°F)
------------------------------------------------------------

                                          Value
Metric                                         
Number of Cold Spells                         0
Average Duration                     0.00 hours
Average Intensity (below threshold)      0.00°F

Abrupt Temperature Changes (hourly diff > 12.66°F)
------------------------------------------------------------

Number of Abrupt Changes Detected: 11
              Timestamp Change (°F)
2025-06-03 08:00:00 PDT       17.16
2025-06-03 09:00:00 PDT       14.10
2025-06-04 14:00:00 PDT       16.44
2025-06-05 11:00:00 PDT       12.99
2025-06-06 08:00:00 PDT       15.22
2025-06-08 08:00:00 PDT       14.13
2025-06-09 08:00:00 PDT       12.90
2025-06-10 08:00:00 PDT       14.14
2025-06-15 11:00:00 PDT       15.22
2025-06-16 08:00:00 PDT       14.33
2025-06-19 08:00:00 PDT       13.81


--- ANALYSIS COMPLETE ---

```
