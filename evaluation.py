import numpy as np


def accuracy(y_pred, y_true):
    """Calculates the accuracy of a model's predictions.
    Accuracy is the number of correct predictions divided by the total number of predictions."""
    return np.sum(y_pred == y_true) / len(y_true)


def precision(y_pred, y_true, positive_label=1):
    """Calculates the precision of a model's predictions.
    Precision is the number of true positive predictions divided by the total number of positive predictions."""
    true_positives = np.sum((y_pred == positive_label) &
                            (y_true == positive_label))
    false_positives = np.sum((y_pred == positive_label)
                             & (y_true != positive_label))
    return true_positives / (true_positives + false_positives)


def recall(y_pred, y_true, positive_label=1):
    """Calculates the recall of a model's predictions.
    Recall is the number of true positive predictions divided by the total number of actual positive instances."""
    true_positives = np.sum((y_pred == positive_label) &
                            (y_true == positive_label))
    false_negatives = np.sum((y_pred != positive_label)
                             & (y_true == positive_label))
    return true_positives / (true_positives + false_negatives)


def f1_score(y_pred, y_true, positive_label=1):
    """Calculates the F1 score of a model's predictions.
    F1 score is the harmonic mean of precision and recall."""
    p = precision(y_pred, y_true, positive_label)
    r = recall(y_pred, y_true, positive_label)
    return 2 * p * r / (p + r)


def confusion_matrix(y_pred, y_true, labels):
    """Calculates the confusion matrix for a model's predictions.
    The confusion matrix is a table of the true positive, true negative, false positive, and false negative predictions for each label."""
    cm = np.zeros((len(labels), 4), dtype=int)
    for i, label in enumerate(labels):
        cm[i] = np.bincount((y_pred == label) & (y_true == label),
                            (y_pred != label) & (y_true != label),
                            (y_pred == label) & (y_true != label),
                            (y_pred != label) & (y_true == label))
    return cm
