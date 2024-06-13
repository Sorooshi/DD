import numpy as np
import os
import pandas as pd
import tensorflow as tf


def splitting_window(features, targets, n_steps, ages=None):
    """
    Splits the features, targets, and optionally ages into windows of size n_steps.

    Args:
        features (np.ndarray): Input features.
        targets (np.ndarray): Input targets.
        n_steps (int): Number of steps in each window.
        ages (np.ndarray, optional): Input ages. Defaults to None.

    Returns:
        np.ndarray: Windowed features.
        np.ndarray: Targets for each window.
        np.ndarray (optional): Windowed ages if ages is not None.
    """
    x_seq, y_seq, age_seq = [], [], []

    for i in range(len(features)):
        end_idx = i + n_steps
        if end_idx >= len(features):
            break
        x_seq.append(features[i:end_idx, :])
        y_seq.append(targets[end_idx])
        if ages is not None:
            age_seq.append(ages[end_idx])

    if ages is not None:
        return np.asarray(x_seq), np.asarray(y_seq), np.asarray(age_seq)
    else:
        return np.asarray(x_seq), np.asarray(y_seq)


def window_dataset_creation(n_steps, path, dataset_name, include_age=False):
    """
    Creates a windowed dataset from the input CSV file.

    Args:
        n_steps (int): Number of steps in each window.
        path (str): Path to the dataset directory.
        dataset_name (str): Name of the dataset file.
        include_age (bool): Whether to include age in the dataset. Defaults to False.

    Returns:
        np.ndarray: Windowed features.
        np.ndarray: Windowed and one-hot encoded targets.
        np.ndarray (optional): Windowed ages if include_age is True.
    """
    #Load the dataset
    fixation = os.path.join(path, dataset_name)
    data = pd.read_csv(fixation)

    df_id = data[data['SubjectID'] == data["SubjectID"].unique()[0]]
    y_train = df_id.loc[:, ["Group"]]
    
    if include_age:
        age_train = df_id.loc[:, ["Age"]]
        x_train = df_id.loc[:, ["FIX_X", "FIX_Y", "FIX_DURATION"]]
        x_train_windowed_final, y_train_windowed_final, age_final = splitting_window(x_train.values, y_train.values, n_steps, age_train.values)
    else:
        x_train = df_id.loc[:, ["FIX_X", "FIX_Y", "FIX_DURATION", "Age"]]
        print(x_train.shape)
        x_train_windowed_final, y_train_windowed_final = splitting_window(x_train.values, y_train.values, n_steps)

    for ids in data["SubjectID"].unique()[1:]:
        df_id = data[data['SubjectID'] == ids]
        y_train = df_id.loc[:, ["Group"]]
        
        if include_age:
            age_train = df_id.loc[:, ["Age"]]
            x_train = df_id.loc[:, ["FIX_X", "FIX_Y", "FIX_DURATION"]]
            x_train_windowed, y_train_windowed, age_windowed = splitting_window(x_train.values, y_train.values, n_steps, age_train.values)
            age_final = np.concatenate((age_final, age_windowed))
        else:
            x_train = df_id.loc[:, ["FIX_X", "FIX_Y", "FIX_DURATION", "Age"]]
            x_train_windowed, y_train_windowed = splitting_window(x_train.values, y_train.values, n_steps)
        
        x_train_windowed_final = np.concatenate((x_train_windowed_final, x_train_windowed))
        y_train_windowed_final = np.concatenate((y_train_windowed_final, y_train_windowed))

    #One-hot encode the targets
    y_train_windowed_final = y_train_windowed_final.reshape(-1) - 1
    y_train_windowed_final = tf.keras.utils.to_categorical(y_train_windowed_final)
    y_train_windowed_final = np.array(y_train_windowed_final)

    if include_age:
        return x_train_windowed_final, y_train_windowed_final, age_final
    else:
        return x_train_windowed_final, y_train_windowed_final