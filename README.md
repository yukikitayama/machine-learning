# Machine Learning

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

## Bagging

- We wanna make a low variance model.
- Suppose we have n independent observations z1, ..., zn, each with variance sigma squared.
- The mean of the observations is z bar.
- The variance of the mean is z bar divided by n
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

## XGBoost

### Big Picture

- Minimize loss function and regularization term

### Detail

- Extreme gradient boosting
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

## Calibration

### Resource

- [How and When to Use a Calibrated Classification Model with scikit-learn](https://machinelearningmastery.com/calibrated-classification-model-in-scikit-learn/)

## K Nearest Neighbor (KNN)

### Resource

- [scikit-learn Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)

## Data

- Imbalance class data
  - [Credit card fraud detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Resource

### Paper

| Topic | Title | Link |
|-------|-------|------|
| XGBoost | XGBoost: A Scalable Tree Boosting System | https://arxiv.org/abs/1603.02754 |

### Book

- An Introduction to Statistical Learning, Springer, Gareth James/Daniela Witten/Trevor Hastie/Robert Tibshirani
- The Elements of Statistical Learning, Springer, Trevor Hastie/Robert Tibshirani/Jerome Friedman

### Video

- [StatQuest](https://www.youtube.com/channel/UCtYLUTtgS3k1Fg4y5tAhLbw)

### Website

- [Machine Learning Mastery](https://machinelearningmastery.com/blog/)

### Data

- [kaggle](https://www.kaggle.com/)
