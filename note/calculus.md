# Calculus

- Select a function which we think might be used to represent the data
- Selecting a function is the creative essence of science.
- The study of how these functions change with respect to their input variables.
- A set of tools for describing the relationship between a function and the changes in its variables.
- `Gradient`
  - A single point as the local gradient, tangent line that touches the curve at a point.
  - e.g. `Acceleration` is the local gradient of a speed-time (y-x) graph
  - Take a continuous function and describe its slope at every point by constructing a new function (its `derivative`)
- `Rise over run`
  - `Gradient = rise / run`
  - How much the straight line function f(x) changes divided by the amount the variable x changes
  - Rise is the amount of change in y, and run is the amount of change in x

![Rise over run](https://github.com/yukikitayama/machine-learning/blob/master/image/rise_over_run.png)

- `Derivative`
  - x is extremely close to 0.
  - It's not zero, because we cannot divide by 0.

![Derivative](https://github.com/yukikitayama/machine-learning/blob/master/image/derivative.png)

## Derivative

- `d/dx cos(x) = -sin(x)`
  - `d/dx sin(x) = cos(x)`
  - `-> d/dx cos(x) = -sin(x)`
  - `-> d/dx -sin(x) = -cos(x)`
  - `-> d/dx -cos(x) = sin(x)`
- `d/dx e^(x) = e^(x)`
- `u(x) = f(x)g(x)h(x), d/dx u(x)?`
  - Use `product rule`
  - Let `A(x) = f(x)g(x), u(x) = A(x)h(x)`
  - `u'(x) = A'(x)h(x) + A(x)h'(x), A'(x) = f'(x)g(x) + f(x)g'(x)`
  - `f'(x)g(x)h(x) + f(x)g'(x)h(x) + f(x)g(x)h'(x)`

### Chain rule

- Useful when the function that we need to take derivative of is a nested function `d/dx A(B(x))`
- Without chain rule, you need to substitute `B(x)` to `A(X)`, and then take derivative of `A(B(x))` w.r.t `x`.
- With chain rule, it's a product of derivative w.r.t each input
  - `d/dx A(B(x)) = d/dB A(B) * d/dx B(x)`

- Multivariate system
  - Calculus to systems with many variables.
  - Use curly partial symbol (d) to differentiate multivariate system
- `Total derivative`
  - Suppose `f(x, y, z)` and `x(t)`, `y(t)`, and `z(t)`
    - x, y, and z are actually all themselves a function of a single other parameter t
  - `df(x, y, z)/dt = df/dx dx/dt + df/dy dy/dt + df/dz dz/dt`
    - The derivative with respect to a new variable `t` is the sum of the chains of the other 3 variables.

## Jacobian

- The Jacobian of a function of many variables `f(x1, x2, x3, ...)` is a row vector where each entry is the partial 
  derivative of `f` w.r.t. each one of those variables in turn.
- If `f(x, y, z)`, then `J = [df/dx, df/dy, df/dz]`
- When we give the Jacobian a specific coordinate, the Jacobian returns a vector pointing in the direction of steepest
  slope of the function.
  - i.e. The Jacobian is a vector that we can calculate for each location of a function which points in the direction of
    the steepest uphill slope.
- The steeper the slope is, the greater the magnitude of Jacobian at that point.
  - When we put the contour lines, large Jacobians exist where the contour lines are tightly packed.
  - The peaks of the mountains and in the bottom of the valleys or on a wide flat plains, Jacobians (gradients) are 
    small.
- Jacobian points uphill.

### Jacobian Matrix

- Jacobian matrix describes functions that take a vector as an input and give a vector as the output.
- Functions are often non-linear but may still be smooth
  - If we zoom in, we can consider each little region of space to be approximately linear
  - Add up all the contributions from the `Jacobian determinants` at each point in space,
  - Calculate the change in the size of a region after transformation.
- Jacobina matrix is a stack of row vectors
  - Each row vector is derivatives w.r.t each variable.

## Hessian

- Get the second order derivatives of a function into a matrix
  - While for the Jacobian we get the first order derivatives of a function into a vector
- If the determinant of Hessian is positive, then we are dealing with either a maximum or a minimum.
  - If the first term in diagonal (top left) is positive, we get a minimum
  - If it's negative, we get a maximum.
- If the Hessian determinant is negative, gradient is 0, and a saddle point.

### Hessian matrix

- `n by n square matrix` where `n` is the number of variables in a function.
- Symmetrical across the leading diagonal.
- Can copy top right triangle region into bottom left.

## Total Derivative

- A multi-variable function `f(x, y, z)` and each variable is a function of `t`.
- The derivative of `f` w.r.t `t` is `df/dt = df/dx dx/dt + df/dy dy/dt + df/dz dz/dt`
- The reason why we do this is because computers are good at this piecewise calculation.
- `df/dt = df/dx . dx/dt`. Derivative of multi-variable function `f` w.r.t `t` is a dot product of two partial 
  derivative vectors.
  - `f`: `f(x) = f(x1, x2, x3, ...)`
  - `.`: Dot product
  - `df/dx`: A vector, `[df/dx1, df/dx2, df/dx3, ...]`
  - `dx/dt`: A vector, `[dx1/dt, dx2/dt, dx3/dt, ...]`


