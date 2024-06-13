import numpy as np


def make_prediction(model, test_dataset):
    """
    Makes predictions on a test dataset using a trained model.

    Parameters:
        model (tf.keras.Model): Trained model to be used for predictions.
        test_dataset (tf.data.Dataset): Dataset containing test data.

    Returns:
        tuple: Predicted labels and true labels.
    """
    fl = False
    y_pred, y_test = None, None
    loaded_model = model
    for x_batch_test, y_batch_test in test_dataset:
        label_proba = loaded_model(x_batch_test, training=False)
        if fl:
            y_pred, y_test = np.concatenate((y_pred, label_proba.numpy()), axis=0), np.concatenate((y_test, y_batch_test.numpy()), axis=0)
        else:
            y_pred, y_test = label_proba.numpy(), y_batch_test.numpy()
            fl = True
    return np.argmax(y_pred, axis=1), np.argmax(y_test, axis=1)