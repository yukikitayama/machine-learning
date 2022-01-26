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
