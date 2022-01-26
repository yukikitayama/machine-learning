## Linear Model

- Why sometimes necessary?
    - When the number of observation is small and/or `p` the number of features is large, a linear model might be all we
      are able to fit to the data without overfitting.

## Regression

- Conditional expectation, the best prediction of Y at any point X = x is the conditional mean
- `Linear basis expansions` is `f(x) = sum of h(x) * theta`, meaning before linear, applying functions or
  transformations to the input vector x.

## Maximum Likelihood Estimation

- Suppose random samples `y` come from a density `P_theta(y)` with parameter `theta`.
- Make an equation of sum of `log(P_theta(y))`
- Assume that the most reasonable value for `theta` are those for which the probability of the observed sample is
  largest.

### Ridge regression

- When there are correlated variables in a linear regression model, a large positive coefficient appear, but it's
  canceled by a similarly large negative coefficient on a correlated variable. It causes high variance. But ridge
  imposes a size constraint on the coefficients, so ridge can alleviate the problem by correlated variables.
- The intercept `Beta_0` is left out of the penalty term.
- `Singular value decomposition (SVD)`
    - If you apply `SVD` to input matrix `X`, then `X = U D V_transposed`.
    - `D` is p * p (p is the number of features) diagonal matrix containing `singular values` of `X`. These singular
      values are scaling factors for ridge shrinkage. If singular value is small, a large shrinkage happens by Ridge.

### Lasso

- Why does Lasso make parameters zero, but Ridge not zero but near zero?
    - Suppose there are only two parameters. RSS has elliptical contours centers at least squares estimate.
    - Ridge has disk, and Lasso has diamond.
    - When the elliptical contour hit the constraint region, because diamond has corners, in Lasso one of the parameters
      is zero, unlike Ridge disk.

![Lasso and ridge](https://github.com/yukikitayama/machine-learning/blob/master/image/esl_3_4_3_discussion_lasso_ridge.png)

### Elastic Net

- `lambda * sum of (alpha * beta^2 + (1 - alpha) * |beta|`
- Elastic Net selects variables like the lasso.
- It also shrinks together the coefficients of correlated predictors like ridge.
