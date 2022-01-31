# Math

## Linear Algebra

- Takes input values and multiplies them by constant (Linear), 
- and have notations describing mathematical objects and manipulates those notations (Algebra).
- Linear algebra is a mathematical system for manipulating vectors in the spaces described by vectors.
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
  - Dot product allows us to find the angle between two vectors.
  - You can change the order of dot product (i.e. uv or vu), because each projection is symmetry, same length projection.

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
  - In physics
    - Ship has velocity `s = [1, 2]`, the current flowing in the direction is `c = [1, 1]`. The velocity of the ship in 
      the direction of the current is the vector projection of `s` onto `c`
    - So `(s.c / |c|^2) * c` is the velocity of the ship in the direction of the current
    - And `s.c / |c|` is the size of the velocity.
- `Triangle inequality`
  - `|a + b| <= |a| + |b|` for every pair of vector `a` and `b`
- `Basis vector`
- Changing basis
- `Linear independence`
  - vector b3 is linearly independent to vectors b1 and b2 if
    - b3 is not equal to a1*b1 + a2*b2 in algebraic way
    - b3 does not lie in the plane spanned by b1 and b2 in a geometric way
- Span
  - If there are `n` vectors in `n` dimensional space, and if they are linearly independent, we say the vectors span the
    space
  - If they are linearly dependent vectors, we can remove one of the two linearly dependent vectors, we have `n - 1`
    vectors in `n` dimensional space, so we say the vectors do not span `n` dimensional space, because it only span
    `n - 1` dimensions.
- Matrix multiplication
  - To get the transformed basis vectors, and to get the multiplication of the vector sum.
  - The multiplying matrix tells us where the basis vectors go.
  - Associative, but not commutative
    - You can start multiplying from any adjacent matrices, but you can't change the order of each matrix.
  - Take every element in each row and multiplied with corresponding element in each column in the other matrix, and 
    adding them up and putting them in place.
- `Inverse matrix`
  - A matrix gives us an identity matrix
  - `A^(-1)A = I`. `A^(-1)` is an inverse matrix of `A`
  - Inverse matrix can be applied from either left or right to still get the identity matrix
- `Triangular matrix`
  - Everything below the body diagonal is zero. 
- `Echelon form`
  - All the numbers below the leading diagonal is zero.
- `Gaussian elimination`
  - A high school way to find parameters
  - Do elimination to get the echelon form triangular matrix
  - Then do back-substitution
  - Then reach the identity matrix to parameter vector to get the answer
  - Gaussian elimination is used to find the inverse matrix. 
    - `A B = I`, apply Gaussian elimination to `A` to make `A` an identity matrix, then `I` becomes `A^(-1)` inverse 
      matrix
- `Singular`
  - A square matrix does not have an inverse matrix
  - Determinant is 0
- `Einstein's summation notation`
  - Allows us to write the sum of each element in row and column without the sigma notation.
    - e.g. `u.v (dot product of two vectors)` is `u_i v_i` in Einstein's summation convention, not 
      `sum from i=0 to n of u_i v_i`
  - Einstein's summation convention makes it easy to see how you calculate it, and how it's going to work.
- Matrix changing basis
  - If vectors are orthogonal, projection (dot product) can give us the new vector.
  - If vectors are not orthogonal, you can solve by matrix transformation, but you cannot use dot product.
- Transformation in a changing basis
  - `A B`. `A` is a A's transformation matrix, and `B` is a B's basis.
  - `A B` is a vector in A's frame.
  - `B^(-1) A B` is a vector back in B's frame.
    - This does the translation from my world to the world of the new basis system.
- `Orthogonal matrix`
- `Orthonormal basis set`
  - A set of unit length basis vectors that are all perpendicular to each other.
  - `a_i.a_j = 0 if i != j (orthogonal) and = 1 if i == j (unit length)`
- `Gram-schmidt process`
  - The goal is to get an orthonormal basis set.
  - Takes a set of vectors and forms an orthonormal basis.
  - Determine the dimension of the space spanned by the basis vectors.