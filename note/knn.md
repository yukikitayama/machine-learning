## K Nearest Neighbor (KNN)

- To predict test data, take K-most similar data from training data are located.
    - Default similarity is found by Euclidean distance.
    - In classification, KNN takes the most common label
    - In regression, KNN takes the average.
- When K is small
    - Classifier is low bias but high variance.
- When K is large
    - Classifier is low variance but high bias.
    - Decision boundary is close to linear
- The training error is always 0 for `K = 1`.
- The effective number of parameters is `N / K`
- Results in large errors if the dimension of the input space is high

### Resource

- ISL 2.2.3
- [Develop k-Nearest Neighbors in Python From Scratch](https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/)
- [scikit-learn Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)
