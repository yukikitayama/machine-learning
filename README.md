# Machine Learning

## Note

- [Linear Discriminant Analysis (LDA)](https://github.com/yukikitayama/machine-learning/blob/master/note/linear_discriminant_analysis.md)
- [Logistic regression](https://github.com/yukikitayama/machine-learning/blob/master/note/logistic_regression.md)
- [Cross-Validation](https://github.com/yukikitayama/machine-learning/blob/master/note/cross_validation.md)
- [Boosting](https://github.com/yukikitayama/machine-learning/blob/master/note/boosting.md)

## Imbalanced Binary Classification

- Precision
  - `TP / (TP + FP)`
  - Means how good a model is at predicting the positive class.
  - Precision doesn't use `TN`.
  - Care about the correct prediction of positive.
  - For example, a precision of 0.33 can be understood as 33% correct predictions among the positive predictions.
- Recall
  - `TP / (TP + FN)`
  - Means how good a model is at predicting the positive class when the actual outcome is positive.
  - `Recall = Sensitivity = True positive rate`
  - Recall doesn't use `TN` too.
  - Care about the correct prediction of positive too.
- False positive rate
  - `FP / (FP + TN)`
  - Means how often a positive class is predicted when the actual outcome is negative.
- True positive rate
  - `TP / (TP + FN)`
- Meaning of precision and recall
  - They don't use the true negatives, only concerned with correctly predicting the positive minority class 1.
- ROC curve
  - X-axis is False positive rate.
  - Y-axis is True positive rate.
  - Means the trade-off between specificity and sensitivity.
  - Left side of x-axis is lower FP and higher TN (Good)
  - Upper side of y-axis is higher TP and lower FN (Good)
  - AUC is the Area Under the ROC Curve.
- Precision-Recall curve
  - X-axis is Recall.
  - Y-axis is Precision.
  - This should be used when there is a moderate to large class imbalance and a large skew in the class distribution.
  - Baseline of precision-recall curve is the proportion of positive class, `P / (P + N)`.
  - PRCAUC is the area under the precision-recall curve.
- Single-threshold measure
  - Scores that need to decide threshold (e.g. 0.5) to assign positive or negative to prediction before calculating 
    scores
  - e.g. Confusion matrix, precision, recall,
- Threshold-free measure
  - The model outputs scores or probabilities for considering positive and negative class, but not a static division.

### SMOTE

- Synthetic Minority Oversampling Technique
  - Oversample the minority class
  - It can balance the class distribution but doesn't provide any additional information on the model

### Resource

- [How to Use ROC Curves and Precision-Recall Curves for Classification in Python](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/)
- [The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/pdf/pone.0118432.pdf)
- [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813)
- [SMOTE for Imbalanced Classification with Python](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/)

## Neural Network

### Big Picture

- Current model makes predictions
- Calculate loss
- Take gradients of the loss
- Update weights

### Loss Functions

- Hinge loss
  - Binary classification
  - Target values are expected to be -1 or 1.
  - In TensorFlow, if you provide 0 or 1 target value, TensorFlow automatically converts to -1 or 1.
  - Output layer should use hyperbolic tangent activation function `Dense(1, activation='tanh')` to convert a single
    value in the range [-1, 1].

### Weight Decay

- ESL Chapter 11
- Penalize by the sum-of-squares of the parameters like Ridge.

### Resource

- [How to Choose Loss Functions When Training Deep Learning Neural Networks](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/)

## Decision Tree

- [Gini index and Entropy](https://github.com/yukikitayama/machine-learning/blob/master/decision-tree/gini_index_entropy.ipynb)

### Pure

- Used in classification with decision tree
- A node is pure when it is predominant with a single class
- Gini index and Entropy both gives us the measurement about how data is uniform or diverse in a node.
  - Entropy has a sharper contrast between uniform group and diverse group, but entropy is a bit more computationally
    expensive than Gini. But fundamentally no difference.

### Gini index

- Used in classification with decision tree
- Small when a node contains predominantly single class observations.
- Large when a leaf node contains a variety of classes.
- A node is pure when Gini index is small.
  - The lower the gini, the easier we can assign the right label for the samples in the group.
  - The lower the gini, the more uniform for the samples in each group.
  - Reducing the Gini allows us to reduce the uncertainty of guessing the group. 

```python
from collections import Counter
from typing import List

def binary_gini(p: float) -> float:
  return p * (1 - p) + (1 - p) * (1 - (1 - p))

def multi_class_gini(data: List[int]) -> float:
  counter = Counter(data)
  proportions = [c / len(data) for c in counter.values()]
  gini = 0
  for proportion in proportions:
    gini += proportion * (1 - proportion)
  return gini
```

#### Gini Gain
  - Reduction of Gini.
  - Measure the quality of the split in decision tree.
  - The higher Gini gain, the better the split

### Entropy

- Used in classification with decision tree
- Small when a node has predominant single class data, and large when it's diverse.
- A node is pure when the entropy is small.

```python
import math
from collections import Counter
from typing import List

def binary_entropy(p: float) -> float:
    return -1 * (p * math.log(p) + (1 - p) * math.log(1 - p))

def multi_class_entropy(data: List[int]) -> float:
    counter = Counter(data)
    proportions = [c / len(data) for c in counter.values()]
    entropy = 0
    for proportion in proportions:
        entropy += proportion * math.log(proportion)
    entropy *= -1
    return entropy
```

#### Information Gain

- Reduction of entropy.

## Bagging

- We wanna make a low variance model.
- Suppose we have n independent observations z1, ..., zn, each with variance sigma squared.
- The mean of the observations is z bar.
- The variance of the mean z bar is sigma squared divided by n
- This variance of mean is smaller than the variance of each observation, because it's divided by n.
- So if we make multiple functions from each training set, and take average of the functions, and then the prediction
  will be low variance model.
- In practice, we don't have multiple training set, so we bootstrap training data to mimic the behaviour.

## Random Forest

- We wanna make a low variance model.
- Bagging doesn't give us a low variance model if the training data contains a very strong predictor.
- In every bootstrap data, the strong predictor is used, the functions made are similar and correlated.
- Taking average of correlated quantities doesn't reduce variance, compared to average of uncorrelated quantities.
- So we sample the predictors to force not to use the strong predictors every time.
- The functions will look different and decorrelated.
- Taking average of those different trees reduce the variance.

## Boosting

- Sequentially apply the weak classification algorithm to repeatedly modified versions of the data.
- Weak classifier is one whose error rate is only slightly better than random guessing.
- The final prediction is a weighted majority vote from all the weak classifiers.
  - The weights give higher influence to the more accurate classifiers in the sequence.
- At step m, observations misclassified by m - 1 model have their weights increased.
  - As iterations proceed, observations difficult to predict receive ever-increasing weight.

## Feature Importance

- It measures, in average, how much RSS (regression) or Gini (classification) are decreased due to splits over a given
  predictor.
  - Big reduction means the feature is important.
- Average over all the trees made by bagged bootstrapped samples.
- In `scikit-learn`, the importance is shown by each feature importance normalized as their sum to 1.

## Gradient Boosting

- Sequentially make tree with about 8 to 32 leaves, not stumps like AdaBoost.
- Make initial guess, average of response
  - In classification, the initial guess is log of odds, log of (number of positive / number of negative)
- Calculate residuals between actual value and the initial guess
  - Pseudo residual
  - This is calculated by taking `gradient of loss function`, but it ends up with residuals.
  - In classification, the residual is between actual response (1 or 0) and the probability made by log of odds into
    logistic function.
- Fit trees with the feature the data and the response the residual.
  - In classification, leaf output value requires transformation.
  - Transformation is `residual / (p * (1 - p))`.
- Scale the predicted residual by a learning rate
- Add the scaled predicted residual to the initial guess
- Repeat

### Resource

- [Gradient Boost Part 3 (of 4): Classification](https://www.youtube.com/watch?v=jxuNLH5dXCs)

## XGBoost (Extreme Gradient Boosting)

### Big Picture

- The objective function to minimize include regularization term as well as loss function to prevent over-fitting.
- Use **shrinkage &eta;** to scale down the influence of each individual tree to prevent over-fitting.
- Use **column subsampling** like random forest to prevent over-fitting.
- Use **quantile** to approximate computationally expensive algorithm to make things efficient.
- Use default direction to allow tree split to be aware data is **sparse** to be good for real world data.

### Detail

- Start from initial guess
- Similarity score
  - Regression: `(sum of residuals, squared) / (number of residuals + lambda)`
  - Classification: `sum of residuals, squared / (sum(prev prob * (1 - prev prob)) + lambda)`
- Gain
- Gamma
- Output value
- Learning rate, eta
  - Scales down the predicted residual to be added to the initial guess
  - Default is 0.3
- Regularization parameter, lambda
- Cover
  - Determine minimum number of residuals in each leaf
  - Denominator of similarity score formula excluding lambda.
  - min_child_weight
  - Default is 1.
- Prune tree
  - Get gain minus gamma
  - If positive, don't prune
  - If negative, prune
- Find output value which minimize loss function and regularization term
- Take derivative with respect to output value
- Take Gradient (1st derivative) and Hessian (2nd derivative) of loss function and regularization term
- In classification
  - Loss function is negative log-likelihood `-[ylog(p) + (1 - y)log(1 - p)]`.
- Approximate greedy algorithm
  - Reduces the number of thresholds that it needs to test to make decision trees to make training fast.
  - Divide data in quantiles, instead of testing all possible thresholds
  - Default is about 33 quantiles.

### Resource

- [XGBoost Part 1 (of 4): Regression](https://www.youtube.com/watch?v=OtD8wVaFm6E)
- [XGBoost Part 2 (of 4): Classification](https://www.youtube.com/watch?v=8b1JEDvenQU)
- [XGBoost Part 3 (of 4): Mathematical Details](https://www.youtube.com/watch?v=ZVFeW798-2I)
- [XGBoost Part 4 (of 4): Crazy Cool Optimizations](https://www.youtube.com/watch?v=oRrKeUCEbq8)

## AdaBoost

### Big Picture

- Make an accurate prediction by weighted summing up many small decision trees.

### Algorithm

- Initialize equal weight `w_i = 1 / N` for each observation.
- Iterate 1 to M
  - Fit decision tree `G_m(x)` to the training data with weights `w_i`.
    - This decision tree classifier is `stump`, two-terminal node classification tree.
  - Compute weighted error rate.
  - Compute weight &alpha;.
  - Rescale weight `w_i` by exp(&alpha;) for observations misclassified by `G_m(x)`.
- Get weighted sum of `G_m(x)`.
- If it classifies, `Discrete AdaBoost`. If it returns a real-valued prediction, it's `Real AdaBoost`.

### Detail

- AdaBoost minimizes `exponential loss criterion` by a forward-stagewise additive modeling.
  - AdaBoost is not optimizing training set misclassification error.
- Use loss function `absolute loss` for regression and `binomial or multinomial deviance loss` for classification.
  - Not `squared-error loss` and `exponential loss`.
- Steepest descent
  - `g`: Gradient of loss function with respect to function
  - `f_m = f_m-1 - scaler * g_m`

### Resource

- [AdaBoost, Clearly Explained](https://www.youtube.com/watch?v=LsK-xG1cLYA)

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

## Calibration

- Match the predicted probabilities with the expected distribution of probabilities for each class.

### Resource

- [How and When to Use a Calibrated Classification Model with scikit-learn](https://machinelearningmastery.com/calibrated-classification-model-in-scikit-learn/)

## Feature Engineering

- Cyclical variables
  - e.g. Month
  - `feature engineered month = sin(month * (2pi / 12))`
  - [Feature Engineering - Handling Cyclical Features](http://blog.davidkaleko.com/feature-engineering-cyclical-features.html)

```python
# Suppose there's Pandas DataFrame df with datetime index
# This date starts from 1, not 0, and goes to 365
df['days_passed_since_new_year'] = [dt.timetuple().tm_yday for dt in df.index.to_pydatetime()]
df['sin_days'] = np.sin((df['days_passed_since_new_year'] - 1) * (2 * np.pi / 365))
```

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

## Clustering

- `Centroid` is a data point representing the center of a cluster.
  - In each kth cluster, take mean in each feature.
- `from sklearn.datasets import make_blobs, make_moons` are convenient functions to generate synthetic clusters.

## K-Means Clustering

- Pre-specify the number of clusters `K`.
- Minimize the sum of the within-cluster variations over all K clusters.
- The most common choice for the within-cluster variation is to use `squared Euclidean distance`.
  - The sum of all the pairwise squared Euclidean distance within kth cluster, divided by the number of data in kth 
    cluster.
- K-mean clustering is called `nondeterministic`, meaning cluster assignment could change depending on the random
  initialization.
  - Commonly run several initializations of the entire k-means algorithm and find the lowest error.
  - By default, `scikit-learn` runs k-means clustering 10 times, and return the one with the lowest error.

### Elbow method

- Choose `K` by seeing the reduction of residual sum of squares

### Silhouette Coefficient

- Values range between -1 and 1.
- Larger numbers indicate samples are closer to their clusters than they are to other clusters.
  - It doesn't work if the clusters are nonspherical.
- `Silhouette coefficient = (b - a) / max(a, b)`
  - `a`: The mean distance between a sample and all other points in the same class.
  - `b`: The mean distance between a sample and all other points in the next nearest cluster.
- `scikit-learn` takes average of silhouette coefficients of all samples into one score
  - Pick the `K` with the max silhouette coefficient.
- [scikit-learn 2.3.10.5. Silhouette Coefficient](https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient)

### Adjusted Rand Index (ARI)

- Evaluate clustering by using both predicted cluster label and true cluster label.
- `sklearn.metrics.adjusted_rand_score` returns close to 0.0 when labeling is random and close to 1.0 when the 
  clusterings are identical.

## DBSCAN

- Density-Based Spatial Clustering of Applications with Noise

## Hierarchical Clustering

- Algorithm
  - Normalize features to make them equally important
  - Make a `dendrogram`, the higher the height, the more the data are different. Horizontal cut defines clusters.
  - Treat each data as its own cluster
  - Fuse the two clusters that are most similar to each other
    - `Euclidean distance` is commonly used to measure similarity.
    - `Linkage` is used within a cluster to calculate single score for similarity.
      - `Average linkage` calculates all pairwise similarity scores within a cluster and take average
      - `Complete linkage` calcuates all pairwise similarity scores and take the largest score.
      - Average and complete linkage tend to yield more balanced clusters.
  - Repeat until all the data belong to one single cluster.
- If the true clusters are not nested, hierarchical clustering could not well represent clusters.
- `Agglomerative clustering`
  - Agglomerate means to collect or gather into a cluster or mass.

### Resource

- [scikit-learn hierarchical clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)

## Dimension Reduction

### PCA

- xxx

### t-SNE

- Maybe pronounce it "tee-snee".
- `t-distributed stochastic neighbor embedding`
- Project data into a low dimensional space so that the clustering in the high dimensional space is preserved.
  - e.g. You can see clusters in 2D scatter plot, apply t-SNE, even in 1D number line, preserve the same clusters.
- [StatQuest: t-SNE, Clearly Explained](https://www.youtube.com/watch?v=NEaUSP4YerM)

### Truncated Singular Value Decomposition (Truncated SVD)

- xxx

## Outlier

- [How to Remove Outliers for Machine Learning](https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/)
- Use interquartile range (IQR)
- Calculate 25th and 75th percentile, multiply a threshold (e.g. 1.5 or 3), and if the data is beyond the threshold, 
  remove the data from the training data
  - The lower the threshold is, the more data are removed as outliers, but it comes with information loss.

## A/B Testing

- Test incremental changes (UX changes, new features, ranking and page load times) to compare pre and post-modification 
  to decide whether the changes are working as desired or not.
- Not good for testing major changes because we can assume that's from something higher than normal engagement or
  emotional responses causing different behavior.
- Methodology
  - Divide data into A (`Control group`) and B (`Test or variant group`) by random sampling.
  - A remains unchanged but implement change in B.
  - Compare the response from A and B to decide which is better
  - Set null hypothesis as no difference between A and B, and alternative hypothesis making changes in B gives us the
    better result.
  - Calculate number for A and B, and run statistical significance test
    - `Type I error`, rejecting null hypothesis when it is true, meaning accept B when B is not better than A.
    - `Type II error`, accept null hypothesis when it is wrong, meaning reject B when B is actually better than A.
    - `Two-sample T-test`, statistical significance to test whether the average difference between the two groups.
    - Set significance level `alpha` like 0.05.

## Bayes Classifier

- Classify to the most probable class using the conditional distribution
- `Bayes rate` is the error rate of Bayes classifier.

## Linear Model

- Why sometimes necessary?
  - When the number of observation is small and/or `p` the number of features is large, a linear model might be all we 
    are able to fit to the data without overfitting.

## Regression

- Conditional expectation, the best prediction of Y at any point X = x is the conditional mean
- `Linear basis expansions` is `f(x) = sum of h(x) * theta`, meaning before linear, applying functions or 
  transformations to the input vector x.

## Maximum Likelihood Estimation

- Suppose random samples `y` come from a density `P_theta(y)` with parameter `theta`.
- Make an equation of sum of `log(P_theta(y))`
- Assume that the most reasonable value for `theta` are those for which the probability of the observed sample is 
  largest.

### Ridge regression

- When there are correlated variables in a linear regression model, a large positive coefficient appear, but it's
  canceled by a similarly large negative coefficient on a correlated variable. It causes high variance. But ridge
  imposes a size constraint on the coefficients, so ridge can alleviate the problem by correlated variables.
- The intercept `Beta_0` is left out of the penalty term.
- `Singular value decomposition (SVD)`
  - If you apply `SVD` to input matrix `X`, then `X = U D V_transposed`. 
  - `D` is p * p (p is the number of features) diagonal matrix containing `singular values` of `X`. These singular 
    values are scaling factors for ridge shrinkage. If singular value is small, a large shrinkage happens by Ridge.

### Lasso

- Why does Lasso make parameters zero, but Ridge not zero but near zero?
  - Suppose there are only two parameters. RSS has elliptical contours centers at least squares estimate.
  - Ridge has disk, and Lasso has diamond.
  - When the elliptical contour hit the constraint region, because diamond has corners, in Lasso one of the parameters 
    is zero, unlike Ridge disk.

![Lasso and ridge](https://github.com/yukikitayama/machine-learning/blob/master/image/esl_3_4_3_discussion_lasso_ridge.png)

### Elastic Net

- `lambda * sum of (alpha * beta^2 + (1 - alpha) * |beta|`
- Elastic Net selects variables like the lasso.
- It also shrinks together the coefficients of correlated predictors like ridge.

## Data

- Imbalance class data
  - [Credit card fraud detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- [Elements of statistical learning data](https://hastie.su.domains/ElemStatLearn/)

## Resource

### Paper

| Topic | Title | Link |
|-------|-------|------|
| XGBoost | XGBoost: A Scalable Tree Boosting System | https://arxiv.org/abs/1603.02754 |

### Book

- An Introduction to Statistical Learning, Springer, Gareth James/Daniela Witten/Trevor Hastie/Robert Tibshirani
- The Elements of Statistical Learning, Springer, Trevor Hastie/Robert Tibshirani/Jerome Friedman
- Mathematical Methods in the Physical Sciences, Mary L Boas
- Introduction to Linear Algebra, Gilbert Strang, Wellesley-Cambridge Press
  - [Introduction to Linear Algebra, Fifth Edition (2016)](https://math.mit.edu/~gs/linearalgebra/)
  - Graduate students textbook

### Video

- [StatQuest](https://www.youtube.com/channel/UCtYLUTtgS3k1Fg4y5tAhLbw)
- [3Blue1Brown](https://www.youtube.com/3blue1brown)
  - Series of videos developing mathematical intuition

### Website

- [Machine Learning Mastery](https://machinelearningmastery.com/blog/)
- [3Blue1Brown](https://www.3blue1brown.com/)
  - Series of videos developing mathematical intuition

### Coursera

- [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)
- [Mathematics for Machine Learning Specialization](https://www.coursera.org/specializations/mathematics-machine-learning)

### Data

- [kaggle](https://www.kaggle.com/)

## Action items

- [x] Review boosting
- [x] Review Gini and Entropy (2021-12-19)
- [x] Review KNN
- [ ] Review calibration
- [ ] Kaggle credit card fraud detection
- [x] Read ISL 10.3.1 K-Means Clustering
- [x] Read ISL 10.3.2 Hierarchical Clustering
- [ ] Read ISL from 9.2 Support Vector Machines
- [x] Read XGBoost paper
- [x] Read SMOTE paper
- [ ] Do Coursera Mathematics for Machine Learning: Linear Algebra, week 2
- [ ] Read ESL from 5.5 Automatic Selection of the Smoothing Parameters
- [ ] Read ESL from 10.10.3 Implementation of Gradient Boosting
- [ ] Check SMOTE
- [ ] Check AB testing
  - https://www.kaggle.com/tammyrotem/ab-tests-with-python
