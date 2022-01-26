## Outlier

- [How to Remove Outliers for Machine Learning](https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/)
- Use interquartile range (IQR)
- Calculate 25th and 75th percentile, multiply a threshold (e.g. 1.5 or 3), and if the data is beyond the threshold,
  remove the data from the training data
    - The lower the threshold is, the more data are removed as outliers, but it comes with information loss.
