## Support Vector Machine (SVM)

- `Support vectors`
    - The data points closest to the hyperplane.
    - Define the separating line by calculating margins.
- `Hyperplane`
    - Divide teh dataset into classes.
- `Margin`
    - The smallest distance between a given separating hyperplane and a data.
    - A gap between the two lines, one line from one side of support vectors, and the other line from the other class
      support vectors.
    - Large margin is good. Small margin is bad.
- `Kernel trick`
    - Function to transform the input space to a higher dimensional space to find a better segregating way.
    - Convert nonseparable problem to separable problem by adding more dimension.
    - `Linear kernel`
    - `Polynomial kernel`
    - `Radial basis function kernel (RBF)`
        - RBF has centroids `mu` and scales `lambda` that have ti be determined.
        - The Gaussian kernel is popular.

### Maximal Margin Classifier

- When the data is separable by a hyperplane, there will exist an infinite number of such hyperplane.
    - Because they can be produced by shifting a bit or rotating a bit.
- `Maxima margin classifier` is a way to define a single classifier among them.
- Find a separating hyperplane which ahs the largest minimum distance to the data.
- Overfits when number of features is many.
- If a separating hyperplane does not exist, then there is no maximal margin classifier, and we need to use `support
  vector classifier` instead by `soft margin`.

### Resource

- [Support Vector Machines with Scikit-learn](https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python)

## Support Vector Machine

- Algorithm
    - Target has +1 or -1.
    - Output negative or positive values depending on which side of the decision boundary it falls.
    - No penalty if an observation is classified correctly and distance from the hyperplane is larger than the margin.
    - Distance from the hyperplane is a measure of confidence.
- Hinge loss function
    - Penalize misclassified data linearly, penalize correctly classified data if low confidence, no penalize the
      correctly classified with confidence.
    - `L(y) = max(0, 1 - t * y)`
        - t is target +1 or -1
        - y is SVM classifier score
    - e.g. t: 1, y: 2, L(y): max(0, 1 - 1 * 2) = 0, classified correctly, and no penalty
    - e.g. t: 1, y: 0.5, L(y): max(0, 1 - 1 * 0.5) = 0.5, classified correctly because y is positive and t is 1, but
      penalty 0.5
    - e.g. t: -1, y: 0.5, L(y): max(0, 1 - (-1) * 0.5) = 1.5, classified incorrectly because y is positive and t is
      negative, and penalty 1.5
- Regularization parameter `C`
    - Scaling the hinge loss.
    - Small `C` means strong regularization, more tolerant misclassification.
    - Large `C` could have over-fitting, try to correctly classify outliers, smaller margin.

### Resource

- [Understanding Hinge Loss and the SVM Cost Function](https://programmathically.com/understanding-hinge-loss-and-the-svm-cost-function/)
