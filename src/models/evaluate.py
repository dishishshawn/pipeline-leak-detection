import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)


def classification_report_df(y_true, y_pred) -> pd.DataFrame:
    """
    Return sklearn classification report as a DataFrame.
    """
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(report).transpose()


def confusion_matrix_df(y_true, y_pred, labels=None) -> pd.DataFrame:
    """
    Return confusion matrix as a labeled DataFrame.
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(
        cm,
        index=[f"actual_{l}" for l in labels],
        columns=[f"pred_{l}" for l in labels],
    )


def roc_auc(y_true, y_score) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute ROC curve and AUC score.

    Returns:
        fpr: false positive rates
        tpr: true positive rates
        auc_score: area under the curve
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score
