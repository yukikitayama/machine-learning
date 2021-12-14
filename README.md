# Machine Learning

## Imbalanced Binary Classification

- Precision
  - `TP / (TP + FP)`, meaning how good a model is at predicting the positive class.
- Recall
  - `TP / (TP + FN)`, meaning
  - `Recall == Sensitivity`
- Meaning of precision and recall
  - They don't use the true negatives, only concerned with correctly predicting the positive minority class 1.
- Precision-Recall curve
  - This should be used when there is a moderate to large class imbalance and a large skew in the class distribution.

### Resource

- [How to Use ROC Curves and Precision-Recall Curves for Classification in Python](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/)
- [The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/pdf/pone.0118432.pdf)

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

## Gradient Boosting

## XGBoost

- Extreme gradient boosting

## K Nearest Neighbor (KNN)

## Data

- Imbalance class data
  - [Credit card fraud detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)