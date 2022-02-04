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