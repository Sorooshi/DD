import os
import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import io

def draw_scatter(X, Y, dur):
    """
    Draws the eye fixation plot plot of fixation points.

    Parameters:
        X (pd.Series): X coordinates of fixation points.
        Y (pd.Series): Y coordinates of fixation points.
        dur (pd.Series): Duration of fixations.

    Returns:
        matplotlib.figure.Figure: The eye fixation plot figure.
    """
    fig, ax = plt.subplots(figsize=(18, 6), dpi=100)
    fig.patch.set_facecolor('black')
    ax.set_axis_off()
    ax.patch.set_facecolor('white')
    ax.patch.set_alpha(0.0)
    ax.invert_yaxis()
    plt.scatter(X, Y, c=dur, s=dur, cmap='gray', vmin=0)
    return fig

def fig_to_np(fig):
    """
    Converts a matplotlib figure to a NumPy array.

    Parameters:
        fig (matplotlib.figure.Figure): The figure to convert.

    Returns:
        np.ndarray: The converted image as a NumPy array.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv.imdecode(img_arr, cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (180, 60), interpolation=cv.INTER_LINEAR)
    img[0, :] = img[img.shape[0] - 1, :] = img[:, 0] = img[:, img.shape[1] - 1] = 0
    plt.close(fig)
    return img


def img_dataset_creation(path, dataset_name):
    """
    Creates a 2D representation.

    Parameters:
        path (str): Path to the dataset directory.
        dataset_name (str): Name of the dataset file.

    Returns:
        tuple: A tuple containing the image dataset and corresponding labels.
    """

    fixation = os.path.join(path, dataset_name)
    fix_data = pd.read_csv(fixation)

    x_train_img, y_train_img = [], []

    for ids in fix_data["SubjectID"].unique():
        X = fix_data[fix_data["SubjectID"] == ids]["FIX_X"]
        Y = fix_data[fix_data["SubjectID"] == ids]["FIX_Y"]
        dur = fix_data[fix_data["SubjectID"] == ids]["FIX_DURATION"]

        img = draw_scatter(X, Y, dur)
        np_img = fig_to_np(img)
        x_train_img.append(np_img)
        y_train_img.append(fix_data[fix_data["SubjectID"] == ids]["Group"].values[0] - 1)

    x_train_img = np.array(x_train_img)
    y_train_img = tf.keras.utils.to_categorical(y_train_img)
    y_train_img = np.array(y_train_img)

    return x_train_img, y_train_img