# Data analysis with pandas

- Pandas is short for "panel data structures"
- Excel for Python

Series
- One-dimensional labelled array
- Combines features of a list and a dictionary
- With built-in functions
  - `sorted(series)` returns sorted Python list.
  - `dict(series)` returns a dictionary with Series index as key and Series value as value.
  - `max(series)` can work
  - `in series` by default searches in Series "index".
- `.iloc[]` is not method, but attribute.
- Search with fallback value
- **Copy** is a duplicate/replica of an object
  - Changes to a copy do not modify the original object
  - `copy` method creates a copy of a pandas object
- **View** is a different way of looking at the same data
  - Changes toa view do modify the original object
- `squeeze()` method gives a view
- `map` method connects (or maps) each Series values to another value.
  - Pass a dictionary or a Series to connect key to value
  - By default, Series value acts as key