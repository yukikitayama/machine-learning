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

DataFrame I: introduction
- Pandas converts a column to float type if an integer column has at least missing value
  - If you wanna convert to integer type, you need to delete or replace the missing value before doing it
- `df.axes` returns `list` of row index series and column index series
- `df.info()`, non-null count means the number of data which are not missing value
- `df.sum(axis="index")` is equal to `df.sum(axis=0)`. `df.sum(axis="columns")` is equal to `df.sum(axis="columns")`
- `df.sum().sum()` can sum all the value in rows and columns
- Pandas extracts a column from a DataFrame as a Series. The Series is a view, so changes to the Series will affect the DataFrame.
- `df[["col1", "col2"]]` creates *copy* of a dataframe, not view.
- `df.dropna()` removes any rows that have any missing values.
  - `dropna(how="all")` doesn't remove row that only all columns are missing
  - `dropna(subset=[])` limit search of columns. *OR* relationship.
- `df["col1"]` is a view, but `df["col1"].fillna(0)` returns a copy
- `astype("category")` type is ideal for columns with a limited number of unique values
  - Reduces memory consumption of dataframe.
  - Pandas doesn't create a separate value in memory for each cell. The cells point to a single copy for each unique value.
  - `df.info()` returns memory usage.
- `df.nunique()` returns a Series with the number of unique values in each column.
- `df.sort_values("col1", na_position="first")` can place NaNs at the beginning.
- `df["col1"].rank(ascending=False)` returns a copy of Series of raking in descending order.

DataFrame II: filtering data
- datetime format `%p` is AM or PM.
- `pd.read_csv("employees.csv", parse_dates=["Start Date"], date_format="%m/%d/%Y")`
- `df[df["col1"] == "value"]` returns a new dataframe.
- `df["datetime_col"].dt.time` returns a Series of `datetime.time`.
- Filter with multiple conditions is better to assign each filter boolean series to a variable to easier to read code.
- `isnull()` returns True for NaN, while `notnull()` return True for present values, so they are invert to another.
- `Series.between(left, right)` returns True if a series value is found within its range.
- `df["col"].duplicated(keep=False)` marks all the occurrence as duplicated if they appear multiple times, unline `keep="first"` or `keep="last"`.
- `drop_duplicates()` by default checks all the columns. `(keep=False)` removes all the duplicated occurrences.
- `df["col"].unique()` returns a different type of data, depending on the column type.

DataFrame III: data extraction
- `df.iloc[start_index:end_index]` end_index ins exclusive, but with `loc`, `df.loc[start_index:end_index]` end_index ins inclusive.








