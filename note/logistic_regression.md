# Logistic Regression

## Fitting Result

- How to interpret the coefficient of logistic regression
  - Suppose coefficient of a predictor is 0.081
  - Take exponential as `exp(0.081) = 1.084 = 8.4%`
  - So 1 unit increase in this predictor increases the `odds` of becoming positive by 8.4%
  - `Odds` is `p / (1 - p)`
- A Z score greater than approximately 2 in absolute value is significant at the 5% level.

## Regularization

- L1 penalty can be used with any linear regression model. For logistic regression, it maximizes the following,

![L1 logistic regression](https://github.com/yukikitayama/machine-learning/blob/master/image/esl_4_4_2_l1_regularized_logistic_regression.png)
ESL 2009 Chapter 4.4.4

## Compared with Linear Discriminant Analysis (LDA)

- Logistic regression and LDA both has a linear function of x; `a_0 + a_k * x`.
  - The linear functions is the form of the log-posterior odds of LDA.
- LDA has the following assumptions, but logistic regression doesn't
  - The class density is the Gaussian
  - The class has a common covariance matrix.
- The way the coefficients are estimated is different.
  - Logistic regression fits the parameters of `P(G|X)` by maximizing the `conditional likelihood`, and the marginal
    density of X `P(X)` just uses the empirical distribution function of the data.
  - LDA fits the parameters by maximizing the full log-likelihood by the `joint density P(X, G)`, and the marginal 
    density `P(X)` plays a role.

## Resource

- ESL 2009 Chapter 4.4 Logistic Regression
- [PennState 9.2.9 - Connection between LDA and logistic regression](https://online.stat.psu.edu/stat508/lesson/9/9.2/9.2.9)
  - I think it's saying the same thing.