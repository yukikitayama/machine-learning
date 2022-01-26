## Feature Engineering

- Cyclical variables
    - e.g. Month
    - `feature engineered month = sin(month * (2pi / 12))`
    - [Feature Engineering - Handling Cyclical Features](http://blog.davidkaleko.com/feature-engineering-cyclical-features.html)

```python
# Suppose there's Pandas DataFrame df with datetime index
# This date starts from 1, not 0, and goes to 365
df['days_passed_since_new_year'] = [dt.timetuple().tm_yday for dt in df.index.to_pydatetime()]
df['sin_days'] = np.sin((df['days_passed_since_new_year'] - 1) * (2 * np.pi / 365))
```
