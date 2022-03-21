# Eigendecomposition

People assume `Eigendecomposition` = `Eigenvalue decomposition` = `Eigenvector decomposition`

## Application

- `Principal component analysis`
- `Regulatization`
  - `Ridge regression`
- `Linear discriminant analysis`
- `Smoothing spline`
- `Support vector machine`

## Concept

- Eigendecomposition is defined only for `square` matrices.
  - While `singular value decomposition` works for any shape
- The goal is to extract 2 sets of features from a matrix; `eigenvalues` and `eigenvectors`
  - A value and a vector is a pair

## Geometric interpretation of eigenvectors/values

- `A v = w` transformation matrix gives an intuition
  - `A`: Transformation matrix
  - `v`: Given vector
  - `w`: Transformed vector
- Typically left-multiplying by `A` gives us a new vector pointing at a different direction and scaled differently.
- But it could produce a new vector pointing at the same direction, but could be with different scale
- In such a pair of transformation matrix and the given vector, the vector is `eigen vector`, and the scale is 
  `eigenvalue`
- In algebra, it can be written in the following
  - `A = lambda` because, even if `A` is a matrix, it doesn't change the direction but could only change the scale, so 
    a matrix can be represented as a scalar.
```
     A v = w
lambda v = A v
```

## Eigenvalue

- When `A v = lambda v` is true, `lambda` is `eigenvalue`, and `v` is the associated `eigenvector`
