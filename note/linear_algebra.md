# Linear Algebra

## Dot Product

- It's also sometimes called `scalar product` because output of dot product is a single number
- Dot product is distributive
  - `a^(T)(b + c) = a^(T)b + a^(T)c` where all are vectors
- Dot product is not associative
  - `a^(T)(b^(T)c) != (a^(T)b)^(T)c`
  - But matrix multiplication is associative
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
- Python
  - `np.multiply(a, b)`
  - `a * b`
  - Where a and b have the same shape.
- Element-wise multiplication is used in joint probability in statistics.

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

## Unit vector

- Dot product of 2 unit vectors is much smaller than dot product of 2 un-normalized vectors
  - Because no scaling from `|a| |b| cos(theta)` and the value is bounded by cosine.
- Used as cosine similarities in machine learning, and pearson correlation coefficient in statistics.

## Field

- Set of numbers where addition, subtraction, multiplication and division are valid
  - Real numbers (R) and complex numbers (C) are field
  - But integers (Z) are not field, because 3 / 2 produces a floating number which does not belong to integers.
- `[3, -4]` can be described as `R^2` meaning `2 dimensional vectors containing real numbers`.

## Subspace

- Set of vectors can be created by addition and scalar multiplication vectors.
  - e.g. If we have a vector `v = [2, 3]`, another vector `u = [4, 6]` is in the same subspace as `v`, because `[4, 6] = 2 * [2, 3] = 2 * v`
- Can be multiple vectors
  - e.g. `v = [2, 3], w = [0, 4]`, `[12, 2]` is in the same subspace because `6v - 4w`
- Subspace must contain the zero vector.
- `Ambient space`
  - Contains subspace
  - e.g. Ambient 3d contains, 0D subspace (Point at the origin), 1D subspace (Line passing the origin), 2D subspace (
    plane passing the origin), 3D subspace (Box containing the origin, same as Ambient 3D space)

## Subset

- A set of points that satisfies some conditions
  - Doesn't need to include the origin
  - Can have boundaries
  - Doesn't need to be addition and scalar multiplication
- Difference between subspace and subset
  - `Subspace` must include the origin, but `subset` doesn't have to
  - `Subspace` is still valid after addition and scalar multiplication, but a point in `subset` could be out of the 
    `subset` after addition and scalar multiplication. 

## Span

- While subspace is a region you can reach by the linear combination of the given vectors.
- And you say those vectors space the subspace.
- Span of set of vectors is all possible linear combinations of all the vectors in the set.
- Frequent question is whether a vector is in the span of another vector or a set of vectors.

## Linear Independence

- Linear in/dependent set is a property of a set of vector, not the property of one of the vectors.
- A set of vector is linearly independent sets if a vector is not a scaled version of another vector, or no vector can
  be expressed by the linear combination of the vectors in the set.
- Format definition of linear dependence is `0 = lambda1 v1 + lambda2 v2 + ... + lambdaN vN, lambda is a member of R (
  real numbers)`
  - Because `lambda1 v1 = lambda2 v2 + ... + lambdaN vN`
  - `v1 = lambda2 / lambda1 v2 + ... lambdaN / lambda1 vN`
  - So v1 is a linear combination of other vectors, so this is linearly dependent set.
- A set of vectors is independent if each vector points in a geometric dimension not reachable using other vectors in 
  the set.
- Any set of M > N vectors in R^N is **dependent**
- Any set of M <= N vectors in R^N *could* be **independent** (Because one could be scaled version of another vector)

## Basis

- `Standard basis vectors`
  - `R^2 [[1, 0], [0, 1]]`
  - `R^3 [[1, 0, 0], [0, 1, 0], [0, 0, 1]]`
- 2 conditions for a set of vectors to be a basis
  1. A set of vectors are linearly independent
  2. A set of vectors span all the subspace.

## Matrix

- By convention, use `M` for rows and `N` for columns
- `MR`. `N`i`C`e guy
- If `M` is a M by N matrix, `C(M)` is a column space of the matrix, and `C(M) is in R^M`, because there are `N` columns
  and each column has `M` elements.
  - `R(M) is in R^N`, row space of a matrix M is in R^N
- `Skew-symmetric`
  - A square matrix which is symmetric across the diagonal, but flips the signs, and diagonal all need to be 0.
  - `A = -A^T`
  - Symmetric matrix is `A = A^T`
- `Shifting a matrix`
  - `A + lambda I = C`
  - Diagonal elements are scaled, but off-diagonal elements do not change.
  - e.g. `[[1, 2], [2, 4]] + 3[[1, 0], [0, 1]] = [[4, 2], [2, 7]]`
- `Hermitian transpose`
  - Transpose for complex matrix.
  - It's to transpose a matrix and to flip signs of complex numbers.

## Diagonal

- In statistics, diagonal elements of a covariance matrix is extracted as a vector to represent the variance of each 
  variable.
- Extracting diagonal elements of a matrix is different from `diagonalizing a matrix`
  - `Diagonalizing a matrix` is a matrix decomposition from eigen decomposition.
- Standard multiplication of diagonal matrix with itself and Hadamard multiplication of diagonal matrix with itself 
  produce the same result, and the result is also diagonal matrix.

## Trace

- Sum of diagonal elements of a matrix
- Trace is only defined in square matrix, while diagonal is defined in any shape of matrix.
- Used in `Frobenius Norm (Frobenius Dot Product)` and `Normalizing eigenvalues or singular values in eigendecomposition
  or singular value decomposition`.

## Broadcasting

- Illegal in traditional linear algebra, but it's often used recently in machine learning
- `[[1, 2], [3, 4]] + [[1], [2]]` is broadcast to `[[1, 2], [3, 4]] + [[1, 1], [2, 2]] = [[2, 3], [5, 6]]`

## Matrix Multiplication

- `AB`
  - A left-multiplies B
  - A pre-multiplies B
  - B right-multiplies A
  - B post-multiplies A

![Four ways to think about matrix multiplication](C:\Users\ykitayama\PycharmProjects\machine-learning\image\matrixMult_4ways.png)

- Matrix multiplication with a diagonal matrix is useful in eigen decomposition and singular value multiplication
  - `AD`, diagonal matrix right-multiplies A, scales A by column by diagonal elements
  - `DA`, diagonal matrix left-multiplies A, scales A by row by diagonal elements
- `Order-of-operations`
  - `(L I V E)^T = E^T V^T I^T LT`
  - Transpose each matrix, flip the order, and then matrix multiplication
  - When applying an operation to multiplied matrices, need to reverse the matrix order.

## Matrix-vector multiplication

- Result is always a vector
  - Resulting vector always has the same orientation as input vector
  - Length of the resulting vector is determined by input matrix
- `Aw` (`A` is matrix. `w` `is vector)
  - Resulting vector is a weighted combinations of the `columns` of `A`
- `w^T A` (`A` is matrix. `w` `is vector)
  - Resulting vector is a weighted combinations of the `rows` of `A`
- When the input matrix is a symmetric matrix, the resulting vector has the same elements.

## Transformation Matrix

- `Matrix @ input vector = output vector`
  - Typically, with this equation, matrix transforms the input vector by rotating and stretching.
  - When the matrix is the `pure rotation matrix`, matrix only transforms the input vector by rotating, no stretching.
    - `[[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]`
  - When it doesn't rotate but only stretches, `eigen things` happen.
    - This phenomenon can be written as `Av = lambda v`, fundamental eigenvalue equation
    - The input vector `v` is `eigenvector` of the matrix
    - The scalar `lambda` is `eigenvalue` of the matrix
    - This matrix multiplication results in the same result by multiplying a vector by a scalar

## Symmetric Matrix

- The symmetric matrix given by `A^T A (or A A^T)` is `covariance matrix` in statistics.
  - Diagonal elements of the symmetric matrix (covariance matrix) is variances in each data
  - Off-diagonal elements are covariance between each data (covariance is un-normalized correlation)
  - The difference from `correlation matrix` is `correlation matrix` is scaled, but `covariance matrix` is not.
- `S = (A + A^T) / 2`
  - Additive method to get symmetric matrix
  - `/ 2` because diagonal element will be double
  - `A` needs to be `square matrix`.
- `A^T A = S (or A A^T = S)`
  - Multiplicative method to get symmetric matrix.

```
[
    [a, b, c],
    [d, e, f]
]
(2 by 3)

@

[
    [a, d],
    [b, e],
    [c, f]
]
(3 by 2)

=

[
    [a^2 + b^2 + c^2, ad + be + cf],
    [ad + be + cf, a^2 + b^2 + c^2]
]
(2 by 2)
```

### Combined Symmetric Matrices

- Sum (Element-wise sum) and Hadamard multiplication (Element-wise multiplication) of two symmetric matrices produces 
  another symmetric matrix
  - Because sum and Hadamard are element-wise operations, so they preserve the input symmetry.
- Matrix multiplication of 2 symmetric matrices does not product a symmetric matrix
  - Because the output matrix is made by different combination of rows and columns from input 2 symmetric matrices.

![Multiplication symmetric matrices](https://github.com/yukikitayama/machine-learning/tree/master/image/multiplication_symmetric_matrices.png)

- But if the input symmetric matrices are `2 by 2 constant diagonal matrix`, multiplication produces another symmetric
  matrix.

## Frobenius Dot Product

- Something between the 2 same shape matrices.
- 3 Methods to compute `frobenius dot product`
  - Element-wise multiplication, and sum all elements.
  - `Vectorize` both matrices, and compute vector dot product.
  - (Most common and computationally efficient way) `tr(A^T B) = <A, B>_F`, taking trace (taking diagonal elements) of 
    `A^T B`.

## Matrix Norm

- `Frobenius norm`, most common measure of matrix magnitude, or `norm`.
  - `norm(A) = sqrt(<A, A>_f) = sqrt(tr(A^T A))`
  - Also called `Euclidean norm`.
- Most commonly used measure of distance of `similarity (Inverse of distance)` between 2 matrices.

## Vectorizing a Matrix

- Concatenate `columns` of a matrix
  - `[[a, c, e], [b, d, e]] = [a, b, c, d, e, f] as column vector`
  - 

## Python

- `np.linalg.norm(VECTOR)` returns vector length.