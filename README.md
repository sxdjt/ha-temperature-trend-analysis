A first attempt at creating rate of change and other statistics, using Home Assitant historical data. 

_Note: The script defaults to pulling the last 30 days of data; my HA integration was reset recently and only has a few days' worth._

```
python ha-temp-trend.py --ha-sensor sensor.patio_climate_temperature --ha-url http://homeassistant.local:8123 --ha-token $BEARER


* Fetching data from Home Assistant for 'sensor.patio_climate_temperature' from 2025-05-12 07:00 to 2025-06-12 06:59 (UTC)...
* Successfully fetched 2686 records from Home Assistant.
* Filtered out 2 non-numeric/unavailable temperature readings.


============================================================
TEMPERATURE DATA ANALYSIS
============================================================

Analysis Period
------------------------------------------------------------

Start: 2025-06-01 10:21:30 PDT
End:   2025-06-11 08:53:57 PDT


Basic Temperature Statistics
------------------------------------------------------------

                   Value (°F)
Metric                       
Mean Temperature        77.52
Median Temperature      77.81
Max Temperature        109.76
Min Temperature         41.90
Standard Deviation      15.66


Trend Analysis Summary
============================================================

               Avg Rate (°F/hour) Avg Duration (hours) Avg Range (°F) Avg R-squared
Trend Type                                                                         
Rising Trends                5.21                 3.03          19.40        0.9326
Cooling Trends              -4.97                 6.15          22.76        0.9346

Heating Degree Days (Baseline 65°F): 8.34
Cooling Degree Days (Baseline 65°F): 96.32

Temporal Patterns
============================================================

Average Temperature by Hour of Day
            Avg. Temp (°F)
hour_of_day               
0                    59.22
1                    57.09
2                    55.27
3                    53.85
4                    52.75
5                    51.30
6                    51.78
7                    57.71
8                    69.31
9                    81.19
10                   87.11
11                   93.32
12                   97.47
13                   99.31
14                   96.85
15                   97.17
16                   94.15
17                   90.35
18                   86.47
19                   81.18
20                   75.49
21                   69.87
22                   65.16
23                   61.75

Average Temperature by Day of Week
------------------------------------------------------------

          Avg. Temp (°F)
Monday             74.17
Tuesday            76.18
Wednesday          62.82
Thursday           73.38
Friday             77.66
Saturday           75.18
Sunday             77.05

Average Daily Temperature Range: 44.45°F
Maximum Daily Temperature Range: 59.04°F


Extreme Events & Anomaly Detection
============================================================

Heat Spells (Temperature > 80°F)
------------------------------------------------------------

                                          Value
Metric                                         
Number of Heat Spells                        10
Average Duration                     9.00 hours
Average Intensity (above threshold)     12.41°F

Cold Spells (Temperature < 40°F)
------------------------------------------------------------

                                          Value
Metric                                         
Number of Cold Spells                         0
Average Duration                     0.00 hours
Average Intensity (below threshold)      0.00°F

Abrupt Temperature Changes (hourly diff > 14.84°F)
------------------------------------------------------------

Number of Abrupt Changes Detected: 4
              Timestamp Change (°F)
2025-06-02 09:00:00 PDT       18.10
2025-06-03 08:00:00 PDT       17.16
2025-06-04 14:00:00 PDT       16.44
2025-06-06 08:00:00 PDT       15.22


--- ANALYSIS COMPLETE ---
```
