# Boosting

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

### Paper

- Use `block structure` and parallel processing to reduce time complexity.
- Use `cache-aware prefetching algorithm` and `block size` make it possible to collect gradient statistics.
- Some data does not fit into main memory, it also stores the data on disk, but to make things fast, it compresses 
  blocks (`Block compression`) and shard the data onto multiple disks (`Block sharding`).
- `Out-of-core computation` and `cache-aware learning` was something new that XGBoost proposed. It allows the limited 
  computing resources to handle large scale data.
- `Weighted quantile sketch` was also new.
- Compared with `scikit-learn` and `R's gbm`, `XGBoost` runs faster and performs at the same accuracy as `scikit-learn`.
- XGBoost can work with terabyte size training data.
- In distributed setting, XGBoost runs faster than `Spark MLLib` and `H2O`.

## Reference

- Paper: XGBoost: A Scalable Tree Boosting System

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
