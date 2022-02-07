# Linear Algebra

## Dot Product

- It's also sometimes called `scalar product` because output of dot product is a single number
- Dot product is distributive
  - `a^(T)(b + c) = a^(T)b + a^(T)c` where all are vectors
- Dot product is not associative
  - `a^(T)(b^(T)c) != (a^(T)b)^(T)c`
  - Bur matrix multiplication is associative!
  - `A(B C) = (A B)C`
- In algebra
  - `sum of a_i * b_i`
- In geometry
  - `cos(theta between a and b) * |a| * |b|`
- `Law of cosines`
  - About how to calculate the third length of a triangle when a triangle is not a right angle triangle, 
  - By two lengths and the angle between them
  - `c^2 = a^2 + b^2 - 2ab cos(theta)`
  - Pythagorean theorem is a special case of law of cosines when theta is 90 degree so cos(90 degree) term will be 0.

## Vector length

- Also called magnitude or norm
- Square root of dot product of the same vector.
- `||a - b||^2 = (a - b)^T (a - b) = ||a||^2 + ||b||^2 - 2 a^T b`
  - Because vector length is a dot product of itself.
  - ^T to make dot product possible for the dimensionality

## Python

- `np.linalg.norm(VECTOR)` returns vector length.