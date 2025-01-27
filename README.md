A first attempt at creating rate of change and other statistics, using Home Assitant historical data.  Basically, I want to see how fast my garage heats and cools.


```
python rate_of_change.py ../history.csv

Mean Temperature: 49.86°F
Median Temperature: 49.46°F
Max Temperature: 61.52°F
Min Temperature: 40.82°F

Average rate of rise: 4.53°F/hour
Average rate of cooling: -1.06°F/hour
Average duration of rising trends: 3.89 hours
Average duration of cooling trends: 5.81 hours
Average temperature range during rise: 9.62°F
Average temperature range during cooling: 6.18°F

R-squared for rising trends: 0.9550
R-squared for cooling trends: 0.9609
```
