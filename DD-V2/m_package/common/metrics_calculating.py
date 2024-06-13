import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def AUC(y_test_hot, y_pred_proba):
    """
    Calculate the Area Under the ROC Curve (AUC) score.

    Args:
        y_test_hot: True binary labels in one-hot encoded form.
        y_pred_proba: Predicted probabilities.

    Returns:
        float: AUC score.
    """
    return roc_auc_score(y_test_hot, y_pred_proba)

def ACC(y_test_labels, y_pred_labels):
    """
    Calculate the accuracy score.

    Args:
        y_test_labels: True binary labels.
        y_pred_labels: Predicted binary labels.

    Returns:
        float: Accuracy score.
    """
    return accuracy_score(y_test_labels, y_pred_labels)

def PR(y_test_labels, y_pred_labels):
    """
    Calculate the precision score.

    Args:
        y_test_labels: True binary labels.
        y_pred_labels: Predicted binary labels.

    Returns:
        float: Precision score.
    """
    return precision_score(y_test_labels, y_pred_labels)

def RECC(y_test_labels, y_pred_labels):
    """
    Calculate the recall score.

    Args:
        y_test_labels: True binary labels.
        y_pred_labels: Predicted binary labels.

    Returns:
        float: Recall score.
    """
    return recall_score(y_test_labels, y_pred_labels)

def F1(y_test_labels, y_pred_labels):
    """
    Calculate the F1 score.

    Args:
        y_test_labels: True binary labels.
        y_pred_labels: Predicted binary labels.

    Returns:
        float: F1 score.
    """
    return f1_score(y_test_labels, y_pred_labels)



def metrics_per_fold_binary(model, test_dataset, metrics_dict):
    """
    Evaluate the model on test dataset and update metrics dictionary for each fold.

    Args:
        model (tf.keras.Model): Trained model.
        test_dataset (tf.data.Dataset): Test dataset.
        metrics_dict (dict): Dictionary to store metrics for each fold.

    Returns:
        dict: Updated metrics dictionary with results from the current fold.
    """
    fl = False
    y_pred, y_test = None, None

    for x_batch_test, y_batch_test in test_dataset:
        label_proba = model(x_batch_test, training=False)

        if fl:
            y_pred, y_test = np.concatenate((y_pred, label_proba.numpy()), axis=0), np.concatenate((y_test, y_batch_test.numpy()), axis=0)
        else:
            y_pred, y_test = label_proba.numpy(), y_batch_test.numpy()
            fl = True

    y_test_lab = np.argmax(y_test, axis=1) 
    y_pred_lab = np.argmax(y_pred, axis=1)

    funcs = [AUC, ACC, PR, RECC, F1] 
    
    for (key, func) in zip(metrics_dict.keys(), funcs):
        if key == "auc_roc":
            metrics_dict[key].append(round(func(y_test, y_pred), 4)) 
        else:
            metrics_dict[key].append(round(func(y_test_lab, y_pred_lab), 4)) 
    return metrics_dict


def resulting_binary(metrics_dict):
    """
    Calculate mean and standard deviation for each metric across all folds.

    Args:
        metrics_dict (dict): Dictionary containing metrics for each fold.

    Returns:
        dict: Dictionary with mean and standard deviation for each metric.
    """
    d = {}
    for k in metrics_dict.keys():
        mean_ = round(np.nanmean(metrics_dict[k]), 4)
        std_ = round(np.std(metrics_dict[k]), 4)
        d[k] = str(mean_) + " pm " + str(std_)
    return d

def linear_per_fold(y_test, y_pred_proba, y_pred_labels, metrics_dict):
    """
    Update metrics dictionary with results from a single fold for not deep models.

    Args:
        y_test: True binary labels.
        y_pred_proba: Predicted probabilities.
        y_pred_labels: Predicted binary labels.
        metrics_dict (dict): Dictionary to store metrics for each fold.

    Returns:
        dict: Updated metrics dictionary with results from the current fold for not deep models.
    """
    funcs = [AUC, ACC, PR, RECC, F1] 
    for (key, func) in zip(metrics_dict.keys(), funcs):
        if key == "auc_roc":
            metrics_dict[key].append(round(func(y_test, y_pred_proba), 4)) 
        else:
            metrics_dict[key].append(round(func(y_test, y_pred_labels), 4)) 
    return metrics_dict
