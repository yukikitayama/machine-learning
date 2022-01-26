## Bagging

- We wanna make a low variance model.
- Suppose we have n independent observations z1, ..., zn, each with variance sigma squared.
- The mean of the observations is z bar.
- The variance of the mean z bar is sigma squared divided by n
- This variance of mean is smaller than the variance of each observation, because it's divided by n.
- So if we make multiple functions from each training set, and take average of the functions, and then the prediction
  will be low variance model.
- In practice, we don't have multiple training set, so we bootstrap training data to mimic the behaviour.

## Random Forest

- We wanna make a low variance model.
- Bagging doesn't give us a low variance model if the training data contains a very strong predictor.
- In every bootstrap data, the strong predictor is used, the functions made are similar and correlated.
- Taking average of correlated quantities doesn't reduce variance, compared to average of uncorrelated quantities.
- So we sample the predictors to force not to use the strong predictors every time.
- The functions will look different and decorrelated.
- Taking average of those different trees reduce the variance.
