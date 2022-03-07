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
  - Square all individual matrix elements, and add them together, and take the square root.
  - Most commonly used measure of distance of `similarity (Inverse of distance)` between 2 matrices.
- `Induced 2-norm (or 2-norm)`
  - How much `A` scales vector `x`
  - `||A||_p = sup(||Ax||_p / ||x||_p), x != 0 vector`
  - If `A` is pure rotation matrix (meaning A is orthogonal matrix), induced 2-norm will be 1, because A doesn't change
    the magnitude
- `Schatten p-norm`
  - Let `sigma` be the singular values of the matrix
  - `||A||_p = ( sum of sigma^p )^(1 / p) from sigma_0 to the last sigma`
  - Sum of all the singular values of the matrix
  - Sum of all singular values is p=1, called the Schatten 1-norm

## Vectorizing a Matrix

- Concatenate `columns` of a matrix
  - `[[a, c, e], [b, d, e]] = [a, b, c, d, e, f] as column vector`

## Self-Adjoint Matrix

- Let `A` be the square symmetric matrix m by m
- Let `v` and `w` be the different vectors with the same size m by 1
- `A` is the `self-adjoint matrix` if `Av . w = v . Aw (dot products)`.
- `(Av)^T w = v^T A^T w = v^T A w`

## Matrix Asymmetry Index

- Asymmetric, or skewed-symmetric.
- `a_i = ||tilda A|| / ||A||`
  - Ratio of norm, `A` is a matrix
  - `tilda A = (A - A^T) / 2`
  - Interpret that when `A` is a symmetric matrix, `tilda A` is 0, so matrix asymmetry index `a_i` is 0.
  - Perfectly skewed-symmetric matrix gives matrix asymmetry index 1.
- Perfectly skewed-symmetric matrix is a matrix whose diagonal elements are 0 and off-diagonal elements are sign flipped
  - e.g. `[[0, 1], [-1, 0]]`
- Symmetric matrix is given by `(A + A^T) / 2`
- Skewed-symmetric matrix is given by `(A - A^T) / 2`

## Rank

- Rank of a matrix is a single number about the amount of the information the matrix has.
- Rank is a non-negative integer, from 0.
- `max(r) = min(m, n)`. Maximum possible rank of a matrix is smaller of number of rows or columns
- Rank is a property of a matrix, not only column or row.
- `Full rank matrix` if A is m by m and rank(A) is m.
- `Full column matrix` if A is m by n, and m > n, and rank(A) is n
- `Full row rank` if A is m by n, and m < n, and rank(A) is m.
- If rank(A) where A is m by n matrix, and rank(A) < m or n
  - `Reduced rank`
  - `Rank deficient`
  - `Degenerate matrix`
  - `Low-rank matrix`
  - `Singular` if A is square matrix
  - `Non-invertable matrix` if A is square matrix
- Rank is dimensionality of information
- Rank of a matrix is the largest number of columns (or rows) that can form a `linearly independent set`
- Rank is used to test whether a matrix has the inverse, because only full rank matrix is invertible.
- Rank is used in PCA, because rank tells us how much information is contained in a large multivariate dataset.

### Computing Rank

- Count the number of columns in a linearly independent set
  - Visual `guess` work for small matrix to know whether it's a linearly independent set
  - Solve `systems of simultaneous linear equations` to check if it's a linearly independent set.
- Apply row reduction to reduce matrix to `echelon form` and count the number of `pivots`.
- Compute the `singular value decomposition` and count the number of `non-zero singular values`.
  - This is the way that software implements and uses to compute rank of a matrix.
- When a matrix is large, rank depends on characteristics of the matrix and `threshold` that software uses, because of
  computing `rounding error`.
- Generally it's safe to assume that a matrix of random numbers haa the maximum possible rank.
  - e.g. `rank(random matrix 4 by 6) = 4`
- `rank(A + B) <= rank(A) + rank(B)`
- `rank(AB) <= min(rank(A), rank(B))`
- `rank(A) = rank(A^T A) = rank(A^T) = rank(A A^T)`
  - `A^T A` spans the same space as `A`
  - `A^T A` has the same dimensionality as `A`
  - `A^T A` has the same singular values (squared) as `A`
- `rank(A) = rank(A^T A)` and if `rank(A)` is `full-rank matrix`, `A^T A` is a `square symmetric full-rank matrix` used 
  in statistics.

### Shifting to Convert Rank-Deficient Matrix to Full-Rank Matrix

- Suppose `A` is a reduced-rank matrix
- Pick a small lambda e.g. `lambda = 0.01`
- Make a full-rank matrix `B = A + lambda * I`
  - `I` is an identity matrix

## Space

### Column Space

- Column space of a matrix is a set of all vectors that can be obtained by any weight linear combination of all the 
  columns in the matrix
- `C(A) = {b_1 a_1 + ... + b_n a_n} = span({a_1, ..., a_n}); b_i are the member of real numbers`
  - `a_i` is a column of a matrix `A`.
  - Span of all the columns
  - `Range` of the columns in matrix A.
- Important question is whether a vector is contained in the column space of a matrix
  - `v <- C(A); <- is notation of contained or member of`
  - `v` and `C(A)` need to be in the same dimension.

### Row Space

- `R(A) = C(A^T)`

### Null Space

- `Trivial way`
  - Multiplying 0 matrix. e.g. `[[3, 2], [5, 4]] @[[0], [0]] = [[0], [0]]`
- `Non-trivial way`
  - Multiplying a vector to matrix in order to make it a 0 vector not in a `trivial way`
  - So not all the elements in the vector are 0. At least 1 element in vector is non-zero.
- `N(A)`, null space of `A`, is the set of all vectors `{v}` such that `Av = 0 vector` and `v != 0 vector`.
- A matrix with a null space has columns forming linearly dependent set
- A matrix with empty null space has columns that form linearly independent set.
- If a vector is not in the null space of a matrix, after multiplying transformation matrix to the vector, you still
  get a non-zero vector
- If a vector is in the null space of a matrix, any transformation matrix multiplication is meaningless, because it
  always leads to 0 vector.
- Eigendecomposition uses `null space` of a matrix to find the eigenvectors of a matrix by computing eigen values and 
  determine a vector which is in the null space of the matrix.
- A vector in null space of a matrix is orthogonal to columns in the matrix, because null space of a matrix is a dot
  product between a vector and columns in a matrix.

### Left-Null Space

- `N(A^T)`
- `A^T v = 0`
- Left-null space `N(A^T)` is orthogonal to the column space of the matrix `C(A)`.
- Dimensionality of column space `C(A)` plus dimensionality of left-null space `N(A^T)` adds up to `ambient` 
  dimensionality.

### Dimension

- When a matrix `A` has `M` rows and `N` columns
  - Column space of the matrix has `M` dimension, because a column is a `M` elements vector
  - `dim(C(A)) + dim(N(A^T)) = M`
    - Dim of column space of `A` plus dim of left-null space of `A` must be `M`
  - `dim(C(A^T)) + dim(N(A^T)) = N`
    - Dim of row space of `A` (because `C(A^T)` is `R(A)`) plus dim of null space of `A` must be `N`
  - These are `orthogonal complements`

## Ax = b

- `Ax = b` in linear algebra
  - `A` is known, `x` is unknown, `b` is known
- `Ax = b` in statistics
  - `X beta = y`
  - `X` is `design matrix`, `beta` is `regression coefficients` vector, `y` is `observed data`.

## Ax = 0

- Meaning of `(A - lambda I) x = 0`
  - `A - lambda I` is shifted version of a matrix
  - `lambda` is `eigenvalue`
  - `x` is `eigenvector`
- Application of `(A - lambda I) x = 0`
  - `Principal component analysis (PCA)`
  - `Generalized eigen decomposition (GED)`
  - `Singular value decomposition (SVD)`
  - `Linear discriminant analysis (LDA)`

## Echelon Form

- `Reduced row echelon form`
  - `RREF` or `rref(A)`
  - All the pivots are 1
  - Elements above, below and to the left of pivots are all 0.
    - Right can be non-zero, but won't be above any pivots
  - Pivots need to be ordered from top left to bottom right

## Pivots

- Elements which have all 0s below and to the left in a matrix.
- Non-zero value
- Number of pivots is `rank` of a matrix

## Gaussian Elimination

- All the pivots are 1
- Augment a matrix with right hand side constant

## Determinant

- `det(A)`
- Only for square matrices
- Single number that reflects the entire matrix.
- If a matrix contains a set of linearly dependent columns, `det(A) = 0`.
- `Determinant` is used to get matrix inverse
  - `Matrix inverse` requires division by determinant
  - So if determinant is 0, there's no inverse.
- Proof of a matrix which has linearly dependent set that has determinant 0.

```
det(A) = 

| a  lambda a |
| c  lambda c |

= a lambda c - lambda a c
```

- `det(I)`, determinant of identity matrix, is always 1 regardless of the size.
- `Singular matrix (reduced-rank matrix)` has a determinant of 0.
  - Even in a determinant of a matrix 3 by 3, if the matrix is a `rank-deficient matrix (reduced-rank matrxi)`, 
    the determinant is 0.
- Computer tends to fail to compute a determinant of 0 in the large size linearly dependent set matrix.
- Exchanging 2 rows in 2 by 2 matrix and 3 by 3 matrix cause the sign flip of the determinant, thought it keeps the same
  magnitude.
  - In 3 by 3 matrix, when exchanging 2 rows twice, it keeps the same sign and magnitude of the determinant.

## Python

- `np.linalg.norm(VECTOR)` returns vector length or called magnitude