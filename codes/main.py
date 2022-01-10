import os
import sys
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf


sys.path.append("../codes")

import data as data
import utilities as util
# import nn_regression as nnr
# import ensemble_regression as enr

tfk = tf.keras

np.set_printoptions(suppress=True, precision=3, linewidth=140)

tf.keras.backend.set_floatx('float32')


def args_parser(args):

    _pp = args.pp.lower()
    _tag = args.tag.lower()
    _run = args.run
    _data_name = args.data_name.lower()
    _note = args.note
    _loss = args.loss.lower()
    _alg_name = args.alg_name.lower()
    _group = args.group
    _project = args.project
    _n_units = args.n_units
    _n_epochs = args.n_epochs
    _optimizer = args.optimizer.lower()
    _batch_size = args.batch_size
    _learning_rate = args.learning_rate
    _n_estimators = args.n_estimators
    _output_dim = args.output_dim
    _n_clusters = args.n_clusters

    return _pp, _tag, _run, _note, _data_name, _loss, _alg_name, _group, _project, _n_units,\
        _n_epochs, _optimizer, _batch_size, _learning_rate, _n_estimators, _output_dim, _n_clusters


def compile_and_fit(model, optimizer, loss, learning_rate, batch_size,
                    n_epochs, x_train, y_train, x_val, y_val, ):

    if optimizer.lower() == "adam":
        model.compile(optimizer=tfk.optimizers.Adam(learning_rate=learning_rate), loss=loss)

    elif optimizer.lower() == "adamax":
        model.compile(optimizer=tfk.optimizers.Adamax(learning_rate=learning_rate), loss=loss)

    elif optimizer.lower() == "rmsprop":
        model.compile(optimizer=tfk.optimizers.RMSprop(learning_rate=learning_rate), loss=loss)

    elif optimizer.lower() == "sgd":
        model.compile(optimizer=tfk.optimizers.SGD(learning_rate=learning_rate), loss=loss)

    else:
        print("undefined optimizer.")

    history = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val),
                        batch_size=batch_size, epochs=n_epochs, verbose=True,)

    return model, history


if __name__ == "__main__":

    # all the string inputs will be converted to lower case.
    parser = argparse.ArgumentParser()

    parser.add_argument("--project", type=str, default="DD",
                        help="Project name for WandB project initialization")

    parser.add_argument("--data_name", type=str, default="DD_Demo",
                        help="Dataset's name, e.g.,"
                             " Dyslexia Detection using: "
                             "1) Demographic, 2) IA_report 3) Fixation_report")

    parser.add_argument("--alg_name", type=str, default="--",
                        help="Name of the algorithm,"
                             " e.g., vnn_reg/dnn_reg/rfr/...")

    parser.add_argument("--run", type=int, default=1,
                        help="Run the model or load the saved"
                             " weights and reproduce the results")

    parser.add_argument("--n_units", type=int, default=6,
                        help="Number of neurons in the"
                             " first hidden layer")

    parser.add_argument("--n_epochs", type=int, default=100,
                        help="Number of epochs")

    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")

    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")

    parser.add_argument("--optimizer", type=str, default="adam",
                        help="Name of the optimizer")

    parser.add_argument("--loss", type=str, default="--",
                        help="Name of the loss function")

    parser.add_argument("--pp", type=str, default=None,
                        help="Data preprocessing method:"
                             " MinMax/Z-Scoring/etc")

    parser.add_argument("--tag", type=str, default="warmup",
                        help="W&B tag will be used to indicate the number of "
                             "run of the same set of experiments to "
                             "compute the average and std.")

    parser.add_argument("--note", type=str, default="--",
                        help="W&B note, e.g., clustering for DD: Demographic")

    parser.add_argument("--group", type=str, default="FClustering",
                        help="W&B group name, i.e., "
                             "using Features: FClustering, FClassification, FRegression, "
                             "or using Time-series: TClustering, TClassification, TRegression")

    parser.add_argument("--n_estimators", type=int, default=100,
                        help="Number of estimators in ensemble regressor algorithms")

    parser.add_argument("--output_dim", type=int, default=2,
                        help="The output dimension of a prediction algorithm")

    parser.add_argument("--n_clusters", type=int, default=2,
                        help="Number of clusters/classes/discrete target values")

    args = parser.parse_args()

    pp, tag, run, note, data_name, loss, alg_name, group, project, n_units, n_epochs, \
        optimizer, batch_size, learning_rate, n_estimators, output_dim, n_clusters = args_parser(args)

    config = {
        "n_estimators": n_estimators,
        "n_clusters": n_clusters,
        "batch_size": batch_size,
        "data_name": data_name,
        "optimizer": optimizer,
        "n_epochs": n_epochs,
        "alg_name": alg_name,
        "lr": learning_rate,
        "n_units": n_units,
        "loss": loss,
        "pp": pp,
      }

    if "nn" in alg_name.lower():
        specifier = alg_name + ", loss=" + loss + ", opt="+optimizer + ", repeat=" + tag + "; "
    elif "ens" in alg_name.lower():
        specifier = alg_name + ", loss=" + loss + ", n_estimators=" + str(n_estimators) + ", repeat=" + tag + "; "
    else:
        specifier = alg_name, ", repeat=" + tag + "; "

    print("specifier:", specifier)

    run = util.init_a_wandb(name=data_name+": "+specifier,
                            project=project,
                            notes=note,
                            group=group,
                            tag=[tag],
                            config=config,
                            )

    data_org, x, y, features, targets, indicators = data.load_data(data_name=data_name, group=group)

    # Preprocessing tha data
    if pp == "rng":
        print("pre-processing:", pp)
        x = util.range_standardizer(x=x)
        y = util.range_standardizer(x=y)
        print("Preprocessed x and y shapes:", x.shape, y.shape)
    elif pp == "zsc":
        print("pre-processing:", pp)
        x = util.zscore_standardizer(x=x)
        y = util.zscore_standardizer(x=y)
        print("Preprocessed x and y shapes:", x.shape, y.shape)
    elif pp == "mm":  # MinMax
        print("pre-processing:", pp)
        x = util.minmax_standardizer(x=x)
        y = util.minmax_standardizer(x=y)
    elif pp == "rs":  # Robust Scaler (subtract median and divide with [q1, q3])
        print("pre-processing:", pp)
        x, rs_x = util.robust_standardizer(x=x)
        y, rs_y = util.robust_standardizer(x=y)
    elif pp == "qtn":  # quantile_transformation with Gaussian distribution as output
        x, qt_x = util.quantile_standardizer(x=x, out_dist="normal")
        y, qt_y = util.quantile_standardizer(x=y, out_dist="normal")
    elif pp == "qtu":  # quantile_transformation with Uniform distribution as output
        x, qt_x = util.quantile_standardizer(x=x, out_dist="uniform")
        y, qt_y = util.quantile_standardizer(x=y, out_dist="uniform")
    elif pp is None:
        x_org = x
        y_org = y
        print("No pre-processing")
    else:
        print("Undefined pre-processing")

    # train, validation and test split:
    train_idx, val_idx, test_idx = util.data_splitter(x=x, validation=True)
    x_train, y_train = x[train_idx, :], y[train_idx, :]
    x_test, y_test = x[test_idx, :], y[test_idx, :]
    x_val, y_val = x[val_idx, :], y[val_idx, :]
    x_test_org, y_test_org = x[test_idx, :], y[test_idx, :]

    print("Data splits shape: \n",
          "\t Train:", x_train.shape, y_train.shape, "\n",
          "\t Val:", x_val.shape, y_val.shape, "\n",
          "\t Test:", x_test.shape, y_test.shape)

    print("************************************************************************")
    print("x_train: \n", x_train[:5, :])
    print("************************************************************************")
    print("y_train: \n", y_train[:5, :])
    print("************************************************************************")

    # start of the program execution
    start = time.time()

    # loss function TF function:
    if loss.lower() == "mae":
        loss_fn = tfk.losses.mean_absolute_error

    elif loss.lower() == "mse":
        loss_fn = tfk.losses.mean_squared_error

    elif loss.lower() == "msle":
        loss_fn = tfk.losses.mean_squared_logarithmic_error

    elif loss.lower() == "mape":
        loss_fn = tfk.losses.mean_absolute_percentage_error

    elif loss.lower() == "kld":
        loss_fn = tfk.losses.kl_divergence

    elif loss.lower() == "cosine_similarity":  # check the loss function here
        loss_fn = tfk.losses.cosine_similarity

    elif loss.lower() == "squared_hinge":  # check the loss function here
        loss_fn = tfk.losses.SquaredHinge

    else:
        print("Loss function is not defined.")

    input_dim = x_train.shape[1]
    # output_dim = y_train.shape[1]

    # instantiating model
    if alg_name.lower() == "vnn_reg":
        _model = nnr.VNNRegression(n_units=n_units, input_dim=input_dim, output_dim=output_dim)

    elif alg_name.lower() == "dnn_reg":
        _model = nnr.DNNRegression(n_units=n_units, input_dim=input_dim, output_dim=output_dim)

    elif alg_name.lower() == "nf_reg":
        model = nf.NFFitter(var_size=output_dim, cond_size=input_dim, batch_size=batch_size,
                            n_epochs=n_epochs, lr=learning_rate)
        model.fit(x_train, y_train)

        history = None

    elif alg_name.lower() == "rfr" or alg_name.lower() == "gbr" or \
            alg_name.lower() == "ar" or alg_name.lower() == "lr":
        model = enr.apply_a_regressor(alg_name=alg_name,
                                      n_estimators=n_estimators,
                                      x_train=x_train, y_train=y_train)
        history = None

    elif alg_name.lower() == "gpr":
        ss_idx = np.random.randint(low=0, high=x_train.shape[0], size=20000)  # because of memory issue
        model = enr.apply_a_regressor(alg_name=alg_name,
                                      n_estimators=n_estimators,
                                      x_train=x_train[ss_idx, :], y_train=y_train[ss_idx, :])
        history = None

    if alg_name.lower() == "vnn_reg_1d":
        _model_iops = nnr.VNNRegression(n_units=n_units, input_dim=input_dim, output_dim=output_dim)
        _model_lat = nnr.VNNRegression(n_units=n_units, input_dim=input_dim, output_dim=output_dim)

    else:
        _model = None
        history = None
        print("Undefined model.")

    if alg_name.lower() == "vnn_reg" or alg_name.lower() == "dnn_reg":
        model, history = compile_and_fit(model=_model, optimizer=optimizer,
                                         loss=loss, learning_rate=learning_rate,
                                         batch_size=batch_size, n_epochs=n_epochs,
                                         x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,)

    # N.B. it could be implemented in a for loop too
    if alg_name.lower() == "vnn_reg_1d" or alg_name.lower() == "dnn_reg_1d":
        model_iops, history_iops = compile_and_fit(model=_model_iops, optimizer=optimizer,
                                                   loss=loss, learning_rate=learning_rate,
                                                   batch_size=batch_size, n_epochs=n_epochs,
                                                   x_train=x_train, y_train=y_train[:, 0],
                                                   x_val=x_val, y_val=y_val[:, 0],)

        model_lat, history_lat = compile_and_fit(model=_model_lat, optimizer=optimizer,
                                                 loss=loss, learning_rate=learning_rate,
                                                 batch_size=batch_size, n_epochs=n_epochs,
                                                 x_train=x_train, y_train=y_train[:, 1],
                                                 x_val=x_val, y_val=y_val[:, 1],)
        history = None

    # plot the train and validation loss function errors
    if history is not None:
        util.plot_loss(run=run, history=history, name=specifier)

    # This part of the code is related to my previous task,
    # in which all the Soroosh_testing data were aggregated together into one array.
    if partitioned is False:
        y_preds = model.predict(x_test)

        # evaluate and saved the result in WandB for the entire Soroosh_testing set
        util.wandb_metrics(run=run, y_trues=y_test, y_preds=y_preds, run_no=x_test.shape[1])

        # plot the predicted values and their std for the entire Soroosh_testing set
        util.wandb_plot_total_predictions(run=run, algorithm=specifier,
                                          y_trues=y_test, y_preds=y_preds)

        # for some datasets this might not work and should be modified accordingly
        # (because of different features location
        features = list(data.columns[2:])
        util.wandb_plot_per_feature_predictions(run=run, features=features,
                                                x_test_org=x_test_org,
                                                x_test=x_test, model=model,
                                                algorithm=specifier, y_trues=y_test, )

        # util.plot_true_pred_distributions(run=run, y_test=y_test, y_pred=y_preds,
        #                                   algorithm=specifier, n_bins=50)

    # This is a very specific Soroosh_testing case scenario which is designed for this project.
    # In this Soroosh_testing scenario each Soroosh_testing run is saved separately
    # from the other as x_test and fitted to trained model.
    elif (partitioned is True and "hdd" in name.lower() or\
          partitioned is True and "ssd" in name.lower()):
        run_no = 0
        meape_iops_mu, meape_lat_mu, gb_mu, qda_mu = [], [], [], []

        for data_test in data_tests:

            run_no += 1

            print("data_tests:", len(data_tests))
            x_test_org, y_test_org = data_test[x_cols].values, data_test[y_cols].values
            if pp == "rng":
                print("pre-processing:", pp)
                x_test = util.range_standardizer_(x_test=x_test_org, x_train=x_org)
                y_test = util.range_standardizer_(x_test=y_test_org, x_train=y_org)
                print("Preprocessed x and y shapes:", x_test.shape, y_test.shape)
            elif pp == "zsc":
                print("pre-processing:", pp)
                x_test = util.zscore_standardizer_(x_test=x_test_org, x_train=x_org)
                y_test = util.zscore_standardizer_(x_test=y_test_org, x_train=y_org)
                print("Preprocessed x and y shapes:", x_test.shape, y_test.shape)
            elif pp == "mm":  # MinMax
                print("pre-processing:", pp)
                x_test = util.minmax_standardizer_(x_test=x_test_org, x_train=x_org)
                y_test = util.minmax_standardizer_(x_test=y_test_org, x_train=y_org)
            elif pp == "rs":  # robust scaler (subtract median and divide with [q1, q3]) >> should be modified later
                print("pre-processing:", pp)
                x_test = util.robust_standardizer_(RS=rs_x, x=x_test_org, )
                y_test = util.robust_standardizer_(RS=rs_y, x=y_test_org, )
            elif pp == "qtn" or pp == "qtu":
                x_test = util.quantile_standardizer_(QT=qt_x, x=x_test_org)
                y_test = util.quantile_standardizer_(QT=qt_y, x=y_test_org)
            elif pp is None:
                print("No pre-processing")
                x_test = x_test_org
                y_test = y_test_org
            else:
                print("Undefined pre-processing")
            if output_dim > 1:
                y_preds = model.predict(x_test)
            elif output_dim == 1:
                y_preds_iops = model_iops.predict(x_test).reshape(-1, 1)
                y_preds_lat = model_lat.predict(x_test).reshape(-1, 1)
                y_preds = np.concatenate((y_preds_iops, y_preds_lat), axis=1)

            print("**************************************************")
            print("x_test: \n", x_test[:5, :])
            print("**************************************************")
            print("y_test: \n", y_test[:5, :])
            print("**************************************************")
            print("y_preds: \n", y_preds[:5])

            eps = np.random.normal(loc=0, scale=10**-6, size=y_preds.shape)
            y_preds = y_preds + eps  # to avoid error when we use QDA

            # plot the predicted values and their std for the entire Soroosh_testing set
            util.wandb_plot_total_predictions(run=run, algorithm=specifier,
                                              y_trues=y_test, y_preds=y_preds, run_no=str(run_no))

            # evaluate and saved the result in WandB for the entire Soroosh_testing set
            _meape_mu, _gb_mu, _qda_mu = util.evaluate_a_x_test(y_trues=y_test, y_preds=y_preds, )

            meape_iops_mu.append(_meape_mu[0])
            meape_lat_mu.append(_meape_mu[1])

            gb_mu.append(_gb_mu)
            qda_mu.append(_qda_mu)

            for io_type in [0, 1]:
                data_io = data_test[data_test['io_type'] == io_type]

                _title = 'Write'
                if io_type == 0: _title = 'Read'
                util.basic_plots(run=run, y_true=y_test, y_pred=y_preds,
                                 run_no=str(run_no), specifier=specifier, title=_title,)

                # if it was need I should modify this function (regarding the feature filtering)
                """
                features = xcols
                util.wandb_plot_per_feature_predictions(run=run, features=features,
                                                        x_test_org=x_test_org,
                                                       x_test=x_test, model=model,
                                                       algorithm=specifier+"-"+str(run_no), y_trues=y_test, )
                """

        meape_iops_mu = np.asarray(meape_iops_mu)
        meape_lat_mu = np.asarray(meape_lat_mu)
        gb_mu = np.asarray(gb_mu)
        qda_mu = np.asarray(qda_mu)

        # save the ave. and std of all Soroosh_testing result in WandB
        util.wandb_metrics(run=run, meape_iops_mu=meape_iops_mu,
                           meape_lat_mu=meape_lat_mu,
                           gb_mu=gb_mu, qda_mu=qda_mu)

    else:
        print("Undefined Soroosh_testing scenario.")

    # end of the program execution
    end = time.time()

    print("Execution time=", end-start)

    if output_dim > 1:
        util.save_model(run=run, model=model, name=alg_name, experiment_name=specifier)
    else:
        util.save_model(run=run, model=model_iops, name=alg_name, experiment_name=specifier + "-iops")
        util.save_model(run=run, model=model_lat, name=alg_name, experiment_name=specifier + "-lat")

    run.finish()




