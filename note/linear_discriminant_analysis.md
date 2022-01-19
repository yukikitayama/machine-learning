# Linear Discriminant Analysis (LDA)

### Concept

- Model each class density as multivariate Gaussian.
- Decision boundaries are linear. Regions are separated by hyperplanes.
- `LDA` is a simple convenient tool, which should be used together with whatever exotic tools.
  - Because it makes simple decision boundaries
  - Because estimates by the Gaussian models are stable
  - Because it has much lower variance than exotic models, although we have a bias of a linear decision boundary.
- Also convenient tool to make multi-class classification.

### Assumption

- The data within each class come from a normal distribution with a class specific mean vector and a common variance.
- Assume that the classes have a common covariance matrix
  - If not use common covariance matrix, we get `quadratic discriminant functions (QDA)`.
  - It means `QDA` requires us to estimate separate covariance matrices for each class.
  
### Modeling

- We don't know the parameters of the Gaussian distributions, so we estimate them using the training data.
  - `pi_k`: Ratio of class-k observations to the training data
  - `mu_k`: Mean of input within class-k
  - `sigma`: Covariance matrix
- The difference between `LDA` and `QDA` appear in the number of parameters
  - `LDA`: `(K - 1) * (p + 1)`, where `p` is the number of features.
  - `QDA`: `(K - 1) * {p * (p + 3) / 2 + 1}`, so it has `p^2` complexity.

## Resource

- ESL 2009 Chapter 4.3 Linear Discriminant Analysis
- [Linear Discriminant Analysis With Python](https://machinelearningmastery.com/linear-discriminant-analysis-with-python/)