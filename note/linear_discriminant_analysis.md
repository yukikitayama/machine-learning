# Linear Discriminant Analysis (LDA)

## Concept

- Model each class density as multivariate Gaussian.
- Decision boundaries are linear. Regions are separated by hyperplanes.
- `LDA` is a simple convenient tool, which should be used together with whatever exotic tools.
  - Because it makes simple decision boundaries
  - Because estimates by the Gaussian models are stable
  - Because it has much lower variance than exotic models, although we have a bias of a linear decision boundary.
- Also convenient tool to make multi-class classification.

## Assumption

- The data within each class come from a normal distribution with a class specific mean vector and a common variance.
- Assume that the classes have a common covariance matrix
  - If not use common covariance matrix, we get `quadratic discriminant functions (QDA)`.
  - It means `QDA` requires us to estimate separate covariance matrices for each class.
  - Covariance matrix is p by p, where p is the number of features.

## Modeling

- We don't know the parameters of the Gaussian distributions, so we estimate them using the training data.
  - `pi_k`: Ratio of class-k observations to the training data
  - `mu_k`: Mean of input within class-k
  - `sigma`: Covariance matrix
- The difference between `LDA` and `QDA` appear in the number of parameters
  - `LDA`: `(K - 1) * (p + 1)`, where `p` is the number of features.
  - `QDA`: `(K - 1) * {p * (p + 3) / 2 + 1}`, so it has `p^2` complexity.
- `Regularized Discriminant Analysis (RDA)` is a weighted average of the separate covariance matrices in `QDA` and the 
  pooled covariance matrix in `LDA`.
  - `Alpha`: the weight

## Benefit

- `K` centroids in `p`-dimensional input space of `LDA` makes a dimension reduction.
  - We can consider the data in a subspace of dimension at most `K - 1`.

## Unclear

- `Canonical (discriminant) variables`
- `Rayleigh quotient`
- `Discriminant coordinates`
- `Canonical variates`

## Resource

- ESL 2009 Chapter 4.3 Linear Discriminant Analysis
- [Linear Discriminant Analysis With Python](https://machinelearningmastery.com/linear-discriminant-analysis-with-python/)