# Math

## Linear Algebra

- Physics and engineering people tend to describe a vector in the bold or normal font, or underline it.
- Math and computer science people don't tend to do that.
- Size of vector
  - Dot itself and take square root
  - `r.r = |r|^2`, then take square root, `sqrt(|r|^2) = |r|`
- `Modulus`
  - The length of a vector
- `Dot product`
  - If dot product is near zero, two vectors are orthogonal
  - If dot product is positive, two vectors are pointing at the same direction
  - If dot product is negative, two vectors are pointing at the opposite direction
  - `r.s = |r||s|cos(theta)`
  - When two vectors are orthogonal, it's 90 degree. cos(90 degree) is 0, so `|r||s| * 0 = 0`, dot product is 0.
  - When two vectors are pointing at the same direction, degree is very small, cos(0) is 1, so `|r||s| * 1` has 
    positive value
  - When two vectors are pointing at the opposite direction, degree is almost 180 degree, cos(180 degree) is -1, so 
    `|r||s| * (-1)` has negative value.

![Projection](https://github.com/yukikitayama/machine-learning/blob/master/image/projection.png)

- `Projection`
  - `Scalar projection` is the size of the above green vector, the vector `s` projected (shadow) onto `r`
  - `Vector projection` is the above green vector itself, the shadow of `s` onto `r`.
  - From `r.s = |r||s|cos(theta)`, `|s|cos(theta)` is projection, shadow of `s` onto `r`.
    - `r.s / |r| = |s|cos(theta)` is called `scalar projection`, because the outcome is a scalar.
    - Scalar project `r.s` over `r` means how much `s` goes along `r`.
    - It's a scalar because `|s|` is a scalar and `cos(theta)` is also a scalar.
    - `r.s / |r|` scalar projection multiplied by `r / |r|` unit vector is `r.s / |r| * r / |r| = (r.s / |r||r|) * r` `vector 
      projection`.
    - It's a vector because `r.s` is a scalar, `|r||r|` is a scalar, so the division is a scalar, but it's multiplied to
      a vector `r`.
  - `Vector projection` of `s` onto `r` is equal to the `scalar projection` of `s` onto `r` multiplied by a vector of
    unit length that points in the same direction as `r`.
    - Because vector projection needs to be a vector, but scalar projection only tells us the size of a vector, so we
      use the unit length of `r` to find the direction.
- `Triangle inequality`
  - `|a + b| <= |a| + |b|` for every pair of vector `a` and `b`