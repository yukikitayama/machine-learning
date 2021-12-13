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

## K Nearest Neighbor (KNN)

## Data

- Imbalance class data
  - [Credit card fraud detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)