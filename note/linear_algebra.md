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
- Dot product of two vectors and the angle relationship

| Dot product | Angle |
|-------------|-------|
| Positive | < 90 degrees (Acute angle) |
| Negative | > 90 degrees (Obtuse angle) |
| 0 | 90 degrees. Orthogonal (Right angle) |
| Product of two vector lengths | 0 degree (No angle) |

## Hadamard Multiplication

- Element-wise multiplication
- Only valid for the two vectors which have the same number of elements.

## Vector Cross-Product

- Only for 2 3-dimensional vectors
- It returns another 3-dimensional vector
- This output vector is orthogonal to the plane defined by the first 2 vectors.
- In multivariate data analysis and machine learning, it's rare to come across cross-product

## Vector length

- Also called magnitude or norm
- Square root of dot product of the same vector.
- `||a - b||^2 = (a - b)^T (a - b) = ||a||^2 + ||b||^2 - 2 a^T b`
  - Because vector length is a dot product of itself.
  - ^T to make dot product possible for the dimensionality

## Cauchy-Schwarz Inequality

- `|a^T b| <= ||a|| ||b||`
  - It says absolute value of dot product between two vector is smaller than or equal to a product of two vector lengths
  - Because `|a^T b| = ||a|| ||b|| |cos(theta)|` and `0 <= |cost(theta)| <= 1`
- The both sides will be equal when angle is 0 degree or 180 degree, because cosine is 1 or -1
- We can also interpret that if a and b are linearly independent, we have `<` relationship
- But if a and b are linearly dependent, i.e. 0 degree or 180 degree, we have `=`.

## Python

- `np.linalg.norm(VECTOR)` returns vector length.