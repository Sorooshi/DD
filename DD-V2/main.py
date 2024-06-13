import argparse
import os
from pathlib import Path
import pickle

import keras
import keras_tuner as kt
from keras_tuner.tuners import BayesianOptimization
import tensorflow as tf
from sklearn.model_selection import train_test_split

from m_package.common.mappings import data_rep_mapping, models_mapping
from m_package.common.metrics_calculating import metrics_per_fold_binary, resulting_binary
from m_package.common.plotting import plot_history, conf_matrix, plot_history_metric_final, plot_history_loss_final
from m_package.common.predictions import make_prediction
from m_package.data.creartion_1D_rep import window_dataset_creation
from m_package.data.creartion_2D_rep import img_dataset_creation
from m_package.data.creartion_3D_rep import DyslexiaVizualization


# Global variables
n_steps = 10
batch_size = 16
num_tune_epochs = 50
num_trials = 20
num_points = 5
dataset_name_ = "Data_binary.csv"


def args_parser(arguments):
    """
    Parse command-line arguments and return relevant values.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        Tuple containing run mode, number of epochs, dataset name in lowercase, model name in lowercase, and type name in lowercase.
    """
    _run = arguments.run
    _epoch_num = arguments.epoch_num
    _data_name = arguments.data_name.lower()
    _model_name = arguments.model_name.lower()
    _type_name = arguments.type_name.lower()

    return  _run, _epoch_num, _data_name, _model_name,  _type_name


def return_optimizer(best_hps):
    """
    Return optimizer based on best hyperparameters.

    Args:
        best_hps: Best hyperparameters from optimization process.

    Returns:
        Optimizer based on hyperparameters.
    """
    optimizer_name = best_hps.values["optimizer"]
    learning_rate = best_hps.values["lr"]

    if optimizer_name == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

    return optimizer


def split_data(X, y):
    """
    Split data into training, validation, and test sets.

    Args:
        X: Input features.
        y: Target labels.

    Returns:
        Datasets for training, validation, and testing.
    """
    X_train, X_valt, y_train, y_valt = train_test_split(X, y, test_size=0.35, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_valt, y_valt, test_size=0.5, stratify=y_valt)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run", type=int, default=2,
        help="Run the model or load the saved"
             " 0 is for creating the datasets"
             " 1 is for tuning"
             " 2 is for training the model"
    )

    parser.add_argument(
        "--epoch_num", type=int, default=5,
        help="Run the model the number of epochs"
    )

    parser.add_argument(
        "--data_name", type=str, default="_huddled",
        help="Dataset's name"
             "_by_size is for time-encoded marker-based representation"
             "_traj is for trajectory tracking using connecting lines representation"
             "_huddled is for multi-level markers representation"
             "_img_fixation is for 2D representation"
             "_windowed is for 1D representation"
    )

    parser.add_argument(
        "--model_name", type=str, default="conv_grad",
        help="Model's name"
            "basic is for shallow neural network"
            "deep is for deeper neural network"
    )

    parser.add_argument(
        "--type_name", type=str, default="conv",
        help="type_name"
            "conv is for convolutional type"
            "lstm is for lstm type"
            "convlstm is for convlstm type"
    )

    args = parser.parse_args()

    run, epoch_num, data_name, model_name, type_name = args_parser(arguments=args)

    data_rep = data_rep_mapping.get(data_name, "1D")

    print(
        "configuration: \n",
        "  Model:", model_name, "\n",
        "  data_name:", data_name, "\n",
        "  run:", run, "\n",
        "  epochs:", epoch_num,"\n",
        "  type", type_name, "\n",
        "  representation type", data_rep, "\n"
    )

    path = Path("Datasets")
    path_tuner = Path("Hyper_params")
    path_figures = Path("Figures")

    #Checking if the run mode on dataset creation
    if run == 0: 
        print("Start of the dataset creation")
        #Creating 2D dataset (eye fixation plots)
        X_img, y_img = img_dataset_creation(path="Datasets", dataset_name=dataset_name_)

        with open(os.path.join(path, f'X_img.txt'),'wb') as f:
            pickle.dump(X_img, f)

        with open(os.path.join(path, f'y_img.txt'),'wb') as f:
            pickle.dump(y_img, f)
        print("Img dataset has been created\n")

        #Multi-level markers dataset
        dataset_creator_huddled = DyslexiaVizualization([32, 64], dataset_name=dataset_name_, path="Datasets", file_format="csv")
        X_h, y_h = dataset_creator_huddled.get_datas("huddle")

        with open(os.path.join(path, f'X_huddled.txt'),'wb') as f:
            pickle.dump(X_h, f)

        with open(os.path.join(path, f'y_huddled.txt'),'wb') as f:
            pickle.dump(y_h, f)
        print("Huddled dataset has been created\n")

        #Trajectory tracking using connecting lines dataset
        dataset_creator_traj = DyslexiaVizualization([16, 64], dataset_name=dataset_name_, path="Datasets", file_format="csv")
        X_t, y_t = dataset_creator_traj.get_datas("traj")

        with open(os.path.join(path, f'X_traj.txt'),'wb') as f:
            pickle.dump(X_t, f)

        with open(os.path.join(path, f'y_traj.txt'),'wb') as f:
            pickle.dump(y_t, f)
        print("Trajectory dataset has been created\n")
        
        #Time-encoded marker-based dataset
        dataset_creator_size = DyslexiaVizualization([16, 64], dataset_name=dataset_name_, path="Datasets", file_format="csv")
        X_s, y_s = dataset_creator_size.get_datas("by_size")

        with open(os.path.join(path, f'X_by_size.txt'),'wb') as f:
            pickle.dump(X_s, f)

        with open(os.path.join(path, f'y_by_size.txt'),'wb') as f:
            pickle.dump(y_s, f)
        print("By size dataset has been created\n")


    #Dataset loading
    if data_name == "_by_size"  and run > 0:
        with open(os.path.join(path, f'X_by_size.txt'),'rb') as f:
            X_data = pickle.load(f)
        with open(os.path.join(path, f'y_by_size.txt'),'rb') as f:
            y_data = pickle.load(f)
        size = [20, 16, 64]
        print("by_size dataset has been loaded")
    elif data_name == "_traj" and run > 0:
        with open(os.path.join(path, f'X_traj.txt'),'rb') as f:
            X_data = pickle.load(f)
        with open(os.path.join(path, f'y_traj.txt'),'rb') as f:
            y_data = pickle.load(f)
        size = [20, 16, 64]
        print("_traj dataset has been loaded")
    elif data_name == "_huddled"  and run > 0:
        with open(os.path.join(path, f'X_huddled.txt'),'rb') as f:
            X_data = pickle.load(f)
        with open(os.path.join(path, f'y_huddled.txt'),'rb') as f:
            y_data = pickle.load(f)
        size = [20, 32, 64]
        print("_huddled dataset has been loaded")
    elif data_name == "_img_fixation"  and run > 0:
        with open(os.path.join(path, f'X_img.txt'),'rb') as f:
            X_data = pickle.load(f)
        with open(os.path.join(path, f'y_img.txt'),'rb') as f:
            y_data = pickle.load(f)
        size = [60, 180]
        print("Img dataset has been loaded")
    elif data_name == "_windowed" and run > 0:
        X_data, y_data = window_dataset_creation(n_steps, path, dataset_name_)
        print("Windowed dataset has been created")


    if run == 1 or run == 2:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:    
            try:
                tf.config.set_logical_device_configuration(
                    device=gpus[0],
                    logical_devices=[
                        tf.config.LogicalDeviceConfiguration(memory_limit=32000)
                    ],
                )
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(e)
        else:
            print("CPU")

    key = (data_rep, model_name, type_name, data_name if data_rep == "3D" else "")
    model_build_func = models_mapping.get(key)
    print(model_build_func)

    proj_name = f'{data_rep}{data_name}_{model_name}_{type_name}'
    model_name_save = f"{data_rep}_{epoch_num}{data_name}_{model_name}_{type_name}"

    if run == 1:
        #Hyperparameter tuning using Bayesian optimization
        train_dataset, val_dataset, test_dataset = split_data(X_data, y_data)
        tuner = BayesianOptimization(
            model_build_func,
            objective=kt.Objective('val_auc', direction='max'),
            max_trials=num_trials,
            num_initial_points=num_points,
            overwrite=True,
            directory='tuning_dir',
            project_name=proj_name)
            
        tuner.search(train_dataset, epochs=num_tune_epochs, validation_data=val_dataset)

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = model_build_func(best_hps)

        with open(os.path.join(path_tuner, proj_name + '.txt'),'wb') as f:
                pickle.dump(best_hps, f)

        #Tuning number of epochs
        train_dataset, val_dataset, test_dataset = split_data(X_data, y_data)
        best_model.compile(optimizer=return_optimizer(best_hps), loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name='auc')])
        history = best_model.fit(train_dataset, validation_data=(val_dataset), epochs=epoch_num)
            
        plot_history(history.history['loss'], history.history['val_auc'], history.history['auc'],  path_figures, model_name_save, history.history['val_loss'])

    if run == 2:
        metrics_results = {
            "auc_roc" : [],
            "accuracy" : [],
            "precision": [],
            "recall": [],
            "f1": []
            }

        with open(os.path.join(path_tuner, proj_name + '.txt'),'rb') as f:
            best_hps = pickle.load(f)

        for key in best_hps.values:
            print(key, best_hps[key])
            
        train_loss, valid_loss = [], []
        train_auc, valid_auc = [], []
        y_true_arr, y_pred_arr = [], []

        #Training and evaluating model for multiple folds
        for i in range(5):
            train_dataset, val_dataset, test_dataset = split_data(X_data, y_data)
            model = model_build_func(best_hps)
            model.compile(optimizer=return_optimizer(best_hps), loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC()])
            history = model.fit(train_dataset, validation_data=(val_dataset), epochs=epoch_num)

            #Collecting training and validation metrics
            train_loss.append(history.history['loss'])
            valid_loss.append(history.history['val_loss'])
            history_keys = list(history.history.keys())
            auc_str = history_keys[1]
            auc_val_str = history_keys[-1]
            train_auc.append(history.history[auc_str])
            valid_auc.append(history.history[auc_val_str])

            #Making predictions and calculating metrics per fold
            y_pred, y_test =  make_prediction(model, test_dataset)
            y_true_arr.append(y_test)
            y_pred_arr.append(y_pred)

            metrics_results = metrics_per_fold_binary(model, test_dataset, metrics_results)

        #Ploting training history and metrics
        plot_history_loss_final(train_loss, valid_loss, "Figures", f"{data_rep}_{epoch_num}{data_name}_{model_name}_{type_name}")
        plot_history_metric_final(train_auc, valid_auc, "Figures", f"{data_rep}_{epoch_num}{data_name}_{model_name}_{type_name}")
        
        #Calculating final evaluation results
        final_results = resulting_binary(metrics_results)
        print(f"RESULTS: for {proj_name}\n")
        print(final_results)

        #Generating confusion matrix
        conf_matrix(y_pred_arr, y_true_arr, f"{model_name_save}")