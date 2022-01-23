# Basis Expansion

- `Cubic spline`
  - Piecewise regression model whose output is continuous, and continuous first and second derivatives at the knots.
  - It is claimed that cubic splines are the lowest-order spline for which the knot-discontinuity is not visible to the 
    human eye.
  - Rare to go beyond cubic-splines, widely orders are M = 1, 2, and 4.
- `Regression spline`
  - Fixed-knot splines.
- `B-spline basis`
  - Efficient computations when the number of knots K is large.
- `Natural cubic spline`
  - Function is linear beyond the boundary knots as constraints.
- For logistic regression
  - Spline is a linear basis expansion in X, so we can first transform X by spline, and use it as the training data for
    fitting logistic regression model
  - The coefficients of logistic regression are multiplied their associated vector of spline basis functions `h_i()`.
