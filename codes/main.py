import os
import sys
import time
import pickle
import argparse
import numpy as np
from sklearn import metrics

sys.path.append("../codes")

import flow as nf
import utilities as util
import nn_regression as nnr
import dyslexia_data as data
import other_regression as otr

np.set_printoptions(suppress=True, precision=3, linewidth=140)


def args_parser(args):

    _pp = args.pp.lower()
    _tag = args.tag.lower()
    _run = args.run
    _data_name = args.data_name  # .lower()
    _note = args.note
    _loss = args.loss.lower()
    _alg_name = args.alg_name  # .lower()
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
    _n_repeats = args.n_repeats

    return _pp, _tag, _run, _note, _data_name, _loss, _alg_name, \
           _group, _project, _n_units,_n_epochs, _optimizer, _batch_size, \
           _learning_rate, _n_estimators, _output_dim, _n_clusters, _n_repeats


if __name__ == "__main__":

    # all the string inputs will be converted to lower case.
    parser = argparse.ArgumentParser()

    parser.add_argument("--project", type=str, default="DD",
                        help="Project name for WandB project initialization.")

    parser.add_argument("--data_name", type=str, default="DD_Demo",
                        help="Dataset's name, e.g.,"
                             " Dyslexia Detection using: "
                             "1) Demographic, 2) IA_report 3) Fixation_report.")

    parser.add_argument("--alg_name", type=str, default="--",
                        help="Name of the algorithm,"
                             " e.g., vnn_reg/dnn_reg/rfr/...")

    parser.add_argument("--run", type=int, default=1,
                        help="Run the model or load the saved"
                             " weights and reproduce the results.")

    parser.add_argument("--n_units", type=int, default=6,
                        help="Number of neurons in the"
                             " first hidden layer.")

    parser.add_argument("--n_epochs", type=int, default=100,
                        help="Number of epochs.")

    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size.")

    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")

    parser.add_argument("--optimizer", type=str, default="adam",
                        help="Name of the optimizer.")

    parser.add_argument("--loss", type=str, default="--",
                        help="Name of the loss function.")

    parser.add_argument("--pp", type=str, default=None,
                        help="Data preprocessing method:"
                             " MinMax/Z-Scoring/etc.")

    parser.add_argument("--tag", type=str, default="warmup",
                        help="W&B tag will be used to filter some of runs"
                             "of the same set of experiments if needed.")

    parser.add_argument("--note", type=str, default="--",
                        help="W&B note, e.g., clustering for DD: Demographic")

    parser.add_argument("--group", type=str, default="FClustering",
                        help="W&B group name, i.e., "
                             "using Features: FClustering, FClassification, FRegression, "
                             "or using Time-series: TClustering, TClassification, TRegression.")

    parser.add_argument("--n_estimators", type=int, default=100,
                        help="Number of estimators in ensemble regressor algorithms.")

    parser.add_argument("--output_dim", type=int, default=1,
                        help="The output dimension of a prediction algorithm.")

    parser.add_argument("--n_clusters", type=int, default=2,
                        help="Number of clusters/classes/discrete target values.")

    parser.add_argument("--n_repeats", type=int, default=10,
                        help="Number of repeats in K-Fold cross validation.")

    args = parser.parse_args()

    pp, tag, run, note, data_name, loss, alg_name, group, \
        project, n_units, n_epochs, optimizer, batch_size, \
        learning_rate, n_estimators, output_dim, n_clusters, n_repeats = args_parser(args)

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

    results = {}
    for repeat in range(1, n_repeats+1):

        repeat = str(repeat)
        results[repeat] = {}

        config = {
            "alg_name": alg_name,
            "n_clusters": n_clusters,
            "n_repeats": n_repeats,
            "repeat": repeat,
            "data_name": data_name,
            "preprocessing": pp,
            "batch_size": batch_size,
            "optimizer": optimizer,
            "n_epochs": n_epochs,
            "lr": learning_rate,
            "n_units": n_units,
            "loss": loss,
            "n_estimators": n_estimators,
        }

        # evaluate and save the result in WandB for the entire test set
        if "regression" in group.lower():
            learning_method = "regression"
        elif "clustering" in group.lower():
            learning_method = "clustering"
        elif "classification" in group.lower():
            learning_method = "classification"
        else:
            print ("Wrong learning method is defined!")
            learning_method = True
            assert learning_method is True

        if "nn" in alg_name.lower():
            specifier = alg_name+", loss="+loss+", opt="+optimizer+", repeat="+repeat

        elif "rfr" in alg_name.lower() or "gbr" in alg_name.lower():
            specifier = alg_name+", n_estimators="+str(n_estimators)+", repeat="+repeat
        else:
            specifier = alg_name+", repeat="+repeat

        print("specifier:", specifier)

        run = util.init_a_wandb(name=data_name+": "+specifier,
                                project=project,
                                notes=note,
                                group=group,
                                tag=[tag],
                                config=config,
                                )

        # train, validation and test split:
        train_idx, val_idx, test_idx = util.data_splitter(x=x, validation=True)
        x_train, y_train = x[train_idx, :], y[train_idx, :].ravel()
        x_test, y_test = x[test_idx, :], y[test_idx, :].ravel()
        x_val, y_val = x[val_idx, :], y[val_idx, :].ravel()
        x_test_org, y_test_org = x[test_idx, :], y[test_idx, :].ravel()

        print("Data splits shape: \n",
              "\t Train:", x_train.shape, y_train.shape, "\n",
              "\t Val:", x_val.shape, y_val.shape, "\n",
              "\t Test:", x_test.shape, y_test.shape)

        print("************************************************************************")
        print("x_train: \n", x_train[:5, :])
        print("************************************************************************")
        print("y_train: \n", y_train[:5])
        print("************************************************************************")

        # start of the program execution
        start = time.time()

        input_dim = x_train.shape[1]
        # output_dim = y_train.shape[1]

        # instantiating model
        if alg_name.lower() == "vnn_reg":
            loss_fn = nnr.determine_tf_loss(loss=loss)
            _model = nnr.VNNRegression(n_units=n_units, input_dim=input_dim, output_dim=output_dim)

        elif alg_name.lower() == "dnn_reg":
            loss_fn = nnr.determine_tf_loss(loss=loss)
            _model = nnr.DNNRegression(n_units=n_units, input_dim=input_dim, output_dim=output_dim)

        elif alg_name.lower() == "nf_reg":
            model = nf.NFFitter(var_size=output_dim, cond_size=input_dim, batch_size=batch_size,
                                n_epochs=n_epochs, lr=learning_rate)
            model.fit(x_train, y_train)

            history = None

        elif alg_name.lower() == "rfr" or alg_name.lower() == "gbr" or \
                alg_name.lower() == "ar" or alg_name.lower() == "lr":
            model = otr.apply_a_regressor(alg_name=alg_name,
                                          n_estimators=n_estimators,
                                          x_train=x_train, y_train=y_train)
            history = None

        elif alg_name.lower() == "gpr":
            # ss_idx = np.random.randint(low=0, high=x_train.shape[0], size=20000)  # because of memory issue
            model = otr.apply_a_regressor(alg_name=alg_name,
                                          n_estimators=n_estimators,
                                          x_train=x_train, y_train=y_train)
            history = None

        else:
            _model = None
            history = None
            print("Undefined model.")

        if alg_name.lower() == "vnn_reg" or alg_name.lower() == "dnn_reg":
            model, history = nnr.compile_and_fit(model=_model, optimizer=optimizer,
                                                 loss=loss, learning_rate=learning_rate,
                                                 batch_size=batch_size, n_epochs=n_epochs,
                                                 x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,)

        # plot the train and validation loss function errors
        if history is not None:
            util.plot_loss(run=run, history=history, name=specifier)

        y_pred = model.predict(x_test).ravel()
        t = np.arange(len(y_test))

        print("Shapes: \n",
              "y_pred", y_pred.shape, "\n",
              "y_test", y_test.shape, "\n"
              "t.shape:", t.shape
              )

        results[repeat]["specifier"] = specifier

        results[repeat]["x_test_org"] = x_test_org
        results[repeat]["y_test_org"] = y_test_org

        results[repeat]["x_test"] = x_test
        results[repeat]["y_test"] = y_test

        results[repeat]["y_pred"] = y_pred

        run = util.wandb_metrics(run=run, y_true=y_test, y_pred=y_pred, learning_method=learning_method)

        # plot the predicted values and their std for the entire test set
        run = util.wandb_true_pred_plots(run=run,
                                         y_true=y_test, y_pred=y_pred,
                                         specifier=specifier,
                                         data_name=data_name,)

        run = util.wandb_true_pred_scatters(run=run,
                                            y_test=y_test, y_pred=y_pred,
                                            specifier=specifier,
                                            data_name=data_name,)

        run = util.wandb_true_pred_histograms(run=run,
                                              y_test=y_test, y_pred=y_pred,
                                              specifier=specifier,
                                              data_name=data_name,
                                              )
        # end of the program execution
        end = time.time()

        print("Execution time of repeat number"+repeat + " is:", end-start)

        run = util.save_model(run=run, model=model, name=alg_name, experiment_name=specifier)

        run.finish()

    with open("../results/"+specifier, "wb") as fp:
        pickle.dump(results, fp)

    # Regression metrics
    MEA, RMSE, MRAE, JSD, R2_Score, MEAPE_mu, MEAPE_std = [], [], [], [], [], [], []
    # Classification and clustering metrics
    ARI, NMI, Precision, Recall, F1_Score, ROC_AUC = [], [], [], [], [], [],

    for repeat, result in results.items():
        y_true = result["y_test"]
        y_pred = result["y_pred"]

        if learning_method == "regression":

            MEA.append(util.mae(y_true=y_true, y_pred=y_pred))
            RMSE.append(util.rmse(y_true=y_true, y_pred=y_pred))
            MRAE.append(util.mrae(y_true=y_true, y_pred=y_pred))
            JSD.append(util.jsd(y_true=y_true, y_pred=y_pred).mean())
            R2_Score.append(metrics.r2_score(y_true, y_pred))
            meape_errors = util.mean_estimation_absolute_percentage_error(
                y_true=y_true, y_pred=y_pred, n_iters=100)

            MEAPE_mu.append(meape_errors.mean(axis=0))
            MEAPE_std.append(meape_errors.std(axis=0))

        else:

            ARI.append(metrics.adjusted_rand_score(y_true, y_pred))
            NMI.append(metrics.normalized_mutual_info_score(y_true, y_pred))
            JSD.append(util.jsd(y_true=y_true, y_pred=y_pred).mean())
            Precision.append(metrics.precision_score(y_true, y_pred))
            Recall.append(metrics.recall_score(y_true, y_pred))
            F1_Score.append(metrics.accuracy_score(y_true, y_pred))
            ROC_AUC.append(metrics.roc_auc(y_true, y_pred))
            meape_errors = util.mean_estimation_absolute_percentage_error(
                y_true=y_true, y_pred=y_pred, n_iters=100)

            MEAPE_mu.append(meape_errors.mean(axis=0))
            MEAPE_std.append(meape_errors.std(axis=0))

    if learning_method == "regression":
        MEA = np.nan_to_num(np.asarray(MEA))
        RMSE = np.nan_to_num(np.asarray(RMSE))
        MRAE = np.nan_to_num(np.asarray(MRAE))
        JSD = np.nan_to_num(np.asarray(JSD))
        R2_Score = np.nan_to_num(np.asarray(R2_Score))
        MEAPE_mu = np.nan_to_num(np.asarray(MEAPE_mu))

        mae_ave = np.mean(MEA, axis=0)
        mae_std = np.std(MEA, axis=0)

        rmse_ave = np.mean(RMSE, axis=0)
        rmse_std = np.std(RMSE, axis=0)

        mrae_ave = np.mean(MRAE, axis=0)
        mrae_std = np.std(MRAE, axis=0)

        jsd_ave = np.mean(JSD, axis=0)
        jsd_std = np.std(JSD, axis=0)

        r2_ave = np.mean(R2_Score, axis=0)
        r2_std = np.std(R2_Score, axis=0)

        meape_ave = np.mean(MEAPE_mu, axis=0)
        meape_std = np.std(MEAPE_mu, axis=0)

        print("     mae  ", " \t rmse ", "\t mrae",
              "\t r2_score ",  "\t meape ", "\t jsd ",
              )

        print(" Ave ", " std", " Ave ", " std ", " Ave ", " std ", " Ave ", " std ",
              " Ave ", " std ", " Ave ", " std ",
              )

        print(
            "%.3f" % mae_ave, "%.3f" % mae_std,
            "%.3f" % rmse_ave, "%.3f" % rmse_std,
            "%.3f" % mrae_ave, "%.3f" % mrae_std,
            "%.3f" % r2_ave, "%.3f" % r2_std,
            "%.3f" % meape_ave, "%.3f" % meape_std,
            "%.3f" % jsd_ave, "%.3f" % jsd_std,
        )
    else:
        JSD = np.nan_to_num(np.asarray(JSD))
        MEAPE_mu = np.nan_to_num(np.asarray(MEAPE_mu))
        ARI = np.nan_to_num(np.asarray(ARI))
        NMI = np.nan_to_num(np.asarray(NMI))
        Precision = np.nan_to_num(np.asarray(Precision))
        Recall = np.nan_to_num(np.asarray(Recall))
        F1_Score = np.nan_to_num(np.asarray(F1_Score))
        ROC_AUC = np.nan_to_num(np.asarray(ROC_AUC))

        ari_ave = np.mean(ARI, axis=0)
        ari_std = np.std(ARI, axis=0)

        nmi_ave = np.mean(NMI, axis=0)
        nmi_std = np.std(NMI, axis=0)

        precision_ave = np.mean(Precision, axis=0)
        precision_std = np.std(Precision, axis=0)

        recall_ave = np.mean(Recall, axis=0)
        recall_std = np.std(Recall, axis=0)

        f1_score_ave = np.mean(F1_Score, axis=0)
        f1_score_std = np.std(F1_Score, axis=0)

        roc_auc_ave = np.mean(ROC_AUC, axis=0)
        roc_auv_std = np.std(ROC_AUC, axis=0)

        jsd_ave = np.mean(JSD, axis=0)
        jsd_std = np.std(JSD, axis=0)

        meape_ave = np.mean(MEAPE_mu, axis=0)
        meape_std = np.std(MEAPE_std, axis=0)

        print("      ari ", " \t nmi ", "\t preci", "\t recall ",
              "\t f1_score ", "\t roc_auc ", "\t meape ", "\t jsd ",
              )

        print(" Ave ", " std", " Ave ", " std ", " Ave ", " std ", " Ave ", " std ",
              " Ave ", " std ", " Ave ", " std ", " Ave ", " std ", " Ave ", " std "
              )

        print("%.3f" % ari_ave, "%.3f" % ari_std,
              "%.3f" % nmi_ave, "%.3f" % nmi_std,
              "%.3f" % precision_ave, "%.3f" % precision_std,
              "%.3f" % recall_ave, "%.3f" % recall_std,
              "%.3f" % f1_score_ave, "%.3f" % f1_score_std,
              "%.3f" % roc_auc_ave, "%.3f" % roc_auv_std,
              "%.3f" % meape_ave, "%.3f" % meape_std,
              "%.3f" % jsd_ave, "%.3f" % jsd_std,

              )





