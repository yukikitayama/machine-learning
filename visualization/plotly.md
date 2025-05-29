# Plotly

- [Plotly Open Source Graphing Library for Python](https://plotly.com/python/)
- Box plot identifies outliers from the median compared to the rest of the data
  - Outliers are values larger (upper outliers) or smaller (lower outliers) than Q3 by at least 1.5 times the IQR.
- *Distplots* is distribution plots. Three layer plots
  - Histogram
  - Rug plot
    - Marks are placed alogn the x-axis for every data point, which lets you see the distribution of values inside each bin
  - Kernel density estimate (KDE)
    - Line that tries to describes the shape of the distribution