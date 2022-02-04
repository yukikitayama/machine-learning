# Math

## Linear Algebra

- Geometric interpretation means interpreting vectors in graph, while algebraic interpretation means interpreting 
  vectors in mathematical notations.

### Vector

- `Dimentionality` of a vector
  - The number of elements.
- Difference between a geometric vector and a coordinate
  - The special case where the vector and the coordinate overlap is when the vector is being drawn in its stardard
    position meaning from origin
  - But vector doesn't need to be from origin all the time.


### Note

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
  - If we see matrix is something to transform a vector, the inverse matrix does `reverse transformation`
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
  - `So long as basis vectors are orthogonal to each other`, transforming a vector to a new coordinate system is just
    taking the project (dot product) of a vector with each of the basis vectors.
- `Orthonormal basis set`
  - A set of unit length basis vectors that are all perpendicular to each other.
  - `a_i.a_j = 0 if i != j (orthogonal) and = 1 if i == j (unit length)`
- `Orthogonal matrix`
  - The matrix composed of `orthonormal basis set`
  - The determinant of an orthogonal matrix is either 1 or -1, because all the basis vectors scale space by a factor of 
    one.
  - `A^(T) = A^(-1)`. If `A^(T)` is the inverse, we can get the identity matrix by `A^(T) A`
  - Rows of the orthogonal matrix are orthonormal
  - Columns of the orthogonal matrix are orthonormal.
  - Transpose matrix of an orthonormal basis vector set is another orthogonal basis vector set
- Why is the orthonormal vector set important?
  - It's important when we transform our data.
  - It means we want our transformation matrix to be an orthogonal matrix
  - It means the inverse is easy to compute.
  - It means transformation is reversible.
  - It means projection is just a dot product (okay to use dot product because basis vectors are orthogonal)
  - It means the determinant is one (after rearranging the basis vectors in right order)
- `Gram-schmidt process`
  - It's about how to get an orthonormal basis set from non-orthogonal and non-unit length vector set.
  - It assumes that 
    1. we have linearly independent vectors that spans the space we are interested in
      - We can check whether the vectors are linearly independent by checking the determinant is not 0.
      - If the vectors are linearly dependent, the determinant is 0.
    2. The vectors are not orthogonal to each other or not of unit length
  - Sequentially takes a vector and forms an orthonormal basis from the previous vectors.
    - Normalize the first vector to be of unit length, which is the first orthonormal vector.
      - The normalization is just `v_1 / |v_1| = e_1` 
    - Get projection of the second vector onto the first unit vector, and subtract this from the second vector, and 
      normalize it. It gives us the second orthonormal vector.
      - The projection is `(v_2 . e_1) e_1`.
      - The subtraction is `v_2 - (v_2 . e_1) e_1 = u_2`
      - The normalization is `u_2 / |u_2| = e_2`
    - Get projection of the third vector onto the first unit vector and the second unit vector, and subtract both from
      the third vector, and normalize it. It gives us the third orthonormal vector.
    - Repeat until all the n vectors are processed.
  - Determine the dimension of the space spanned by the basis vectors.
- `Eigen-`
  - Means characteristic
  - We take a transform, and we look for the vectors who are still laying on the same span as before, and then we 
    measure how much their length has changed.
- `Eigenvalue`
  - Value of eigenvectors. If after transformation, the length doesn't change, the eigenvalue is 1.
  - If after transformation, eigenvector length doubled, eigenvalue is 2.
  - The amount that each eigenvector has been stretched in a linear transformation.
- `Eigenvector`
  - When transforming (scaling, rotations, and shears), some vectors end up lying on the same line (Same direction and
    same length), while any other vectors changed
  - These unchanged vectors are special. They are characteristic of a particular transformation
  - People call them eigenvectors.
  - Vectors which lie along the same span both before and after applying a linear transform to a space.
- `Pure shear`
  - No scaling, no rotation, and are unchanged
- Special eigen-cases
  1. Uniform scaling
    - Scale by the same amount in each direction
    - Any vector is an eigenvector
  2. 180 degrees rotation
    - Stay the same span but pointing in the opposite direction
    - All vectors are eigenvectors, but eigenvalues are -1.
  3. Horizontal shear and vertical scaling
    - Horizontal vector is an eigenvector and its eigenvalue is 1.
    - There is another vector which is eigenvector but not easy to spot.
- Calculating eigenvectors
  - `Ax = lambda * x`
    - `A`: Transformation matrix, n dimensional transform, n by n square matrix
    - `lambda`: Scalar, not matrix or vector
    - `x`: Eigenvector, n dimensional vector
  - Goal is to find values of `x` that make the two sides equal
  - `(A - lambda I)x = 0`
  - Take determinant of `A - lambda I` to test if a matrix operation will result in 0.
  - `det(A - lambda I) = 0`
- `Characteristic polynomial`
  - Evaluation of `det(A - lambda I) = 0`
  - Let A be `[[a, b], [c, d]]`
  - `det([[a, b], [c, d]] - [[lambda, 0], [0, lambda]])`
  - `det([[a - lambda, b], [c, d - lambda]])`
  - `(a - lambda)(d - lambda) - bc` because of [2 by 2 matrix determinant equation](https://en.wikipedia.org/wiki/Determinant#2_%C3%97_2_matrices)
  - `ad - a lambda - d lambda + lambda^2 - bc = 0`
  - `lambda^2 - (a + d)lambda + (ad - bc) = 0`. This is the `characteristic polynomial`.
  - Eigenvalues are the solutions of the above equation.
  - Eigenvectors are found by plugging the solved eigenvalues into the original expression `(A - lambda I)x = 0`
- `Eigenbasis`
  - Change a matrix to a basis where transformation `T` becomes diagonal
  - People do this because diagonal matrix is easy to do n-power calculation.
- `Diagonal matrix`
  - `D = C^(-1) T C`
    - `C`: Eigenvectors. First column of `C` is the first eigenvector, and so on.
    - `C^(-1)`: Inverse of eigenvectors
    - `T`: Transformation matrix
    - `D`: Diagonal matrix. Each element in diagonal is eigenvalue, `[[lambda_1, 0], [0, lambda_2]]`
  - `T = C D C^(-1)`
  - `T^n = C D^n C(-1)`
- `PageRank`
  - The ranking of websites by find the probability that a person will be at the end of a certain process
  - Decide which order to display the websites when they returned from search.
  - Assume that the importance of a website is related to its links to and from other websites
  - `Power iteration method`
    - Repeatedly multiplying a randomly selected initial guess vector by a matrix
    - Good at dealing with large system.
  - `Dumping factor, d`
    - Probability that someone randomly types in a web address, rather than clicking on a link on the current page.
    - Add a small probability that a person does not follow any link on a webpage and visit another at random
    - This is to avoid stuck in a website because it doesn't have outgoing link, which disrupts PageRank calculation.
    - `M = d L + (1 - d) / n * J`
      - `J`: n by n matrix where every element is one
      - `n`: Number of websites
      - If `d` is 1, iteration always use transformation matrix `L`
      - If `d` is 0, `L` disappear, and always visit a website at random because `1 / n`
  - `L r = r`
    - Consider this equation as `L` is a transformation matrix, and eigenvalue `1` (multiplied to right-hand side)
    - `L`: Link matrix. Matrix which has columns which represent the probability of leaving a website for any other websites, sum to
      one. It has eigenvalue 1.
    - `r`: A vector that eventually contains the probabilities for each website that a person stay there.

## Calculus

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