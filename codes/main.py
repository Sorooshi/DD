import os
import sys
import time
import pickle
import argparse
import numpy as np
import baseline as bl
import utilities as util
import regressions as reg
import clusterings as clu
import dyslexia_data as data
import classifications as cls


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
    _target_is_org = args.target_is_org

    return _pp, _tag, _run, _note, _data_name, _loss, _alg_name, \
           _group, _project, _n_units,_n_epochs, _optimizer, _batch_size, \
           _learning_rate, _n_estimators, _output_dim, _n_clusters, _n_repeats, _target_is_org


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

    parser.add_argument("--n_epochs", type=int, default=1000,
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

    parser.add_argument("--target_is_org", type=int, default=1,
                        help="Weather to use not preprocessed target values or not.")

    args = parser.parse_args()

    pp, tag, run, note, data_name, loss, alg_name, group, \
        project, n_units, n_epochs, optimizer, batch_size, \
        learning_rate, n_estimators, output_dim, n_clusters, n_repeats, target_is_org = args_parser(args)

    # evaluate and save the result in WandB for the entire test set
    if "regression" in group.lower():
        learning_method = "regression"
    elif "clustering" in group.lower():
        learning_method = "clustering"
    elif "classification" in group.lower():
        learning_method = "classification"
    elif "baseline" in group.lower():
        learning_method = "baseline"
    else:
        print ("Wrong learning method is defined!")
        learning_method = True
        assert learning_method is True

    if run == 1:

        data_org, x, y, features, targets, indicators = data.load_data(data_name=data_name, group=group)

        # Preprocessing tha data
        x_org, y_org = x, y
        x, y = data.preprocess_data(x=x, y=y, pp=pp)

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

            if "nn" in alg_name.lower():
                specifier = alg_name+", loss="+loss+", opt="+optimizer + \
                            ", repeat="+repeat+", pp="+pp+" target_is_org="+str(target_is_org)

            elif "rf" in alg_name.lower() or "gb" in alg_name.lower():
                specifier = alg_name+", n_estimators="+str(n_estimators) + \
                            ", repeat="+repeat+", pp="+pp+" target_is_org="+str(target_is_org)
            else:
                specifier = alg_name+", repeat="+repeat+", pp="+pp + \
                            " target_is_org="+str(target_is_org)

            print("specifier:", specifier)

            run = util.init_a_wandb(name=data_name+": "+specifier,
                                    project=project,
                                    notes=note,
                                    group=group,
                                    tag=[tag],
                                    config=config,
                                    )

            # train, validation and test split:
            x_train, y_train, x_val, y_val, x_test, y_test, = data.data_splitter(
                x=x, y=y, x_org=x_org, y_org=y_org, target_is_org=target_is_org,
            )

            # start of the program execution
            start = time.time()

            input_dim = x_train.shape[1]
            # output_dim = y_train.shape[1]

            # instantiating and fitting the model
            if learning_method == "regression":
                model, history = reg.instantiate_fit_reg_model(
                    alg_name=alg_name, loss=loss, n_units=n_units,
                    input_dim=input_dim, output_dim=output_dim,
                    batch_size=batch_size, n_epochs=n_epochs,
                    learning_rate=learning_rate,
                    x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                    n_estimators=n_estimators, optimizer=optimizer,
                )

            elif learning_method == "clustering":
                model, history = clu.instantiate_fit_clu_model(
                    alg_name=alg_name,
                    n_clusters=n_clusters,
                    x_train=x_train, y_train=y_train
                )

            elif learning_method == "classification":
                model, history = cls.instantiate_fit_cls_model(
                    alg_name=alg_name, loss=loss, n_units=n_units,
                    input_dim=input_dim, output_dim=output_dim,
                    batch_size=batch_size, n_epochs=n_epochs,
                    learning_rate=learning_rate,
                    x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                    n_estimators=n_estimators, optimizer=optimizer,
                )

            elif learning_method == "baseline":
                model, history, y_pred = bl.instantiate_fit_baseline_model(
                    y_train=y_train, y_test=y_test, target_is_org=target_is_org
                )

            # plot the train and validation loss function errors
            if history is not None:
                util.plot_loss(run=run, history=history, name=specifier)

            if learning_method != "baseline":
                try:
                    y_pred = model.predict(x_test).ravel()
                except:
                    y_pred = model.fit_predict(x_test).ravel()

            # to avoid 1) ill-defined situation in computing precision, recall, f1_score and roc_auc,
            # 2) to avoid wrong computation in MEAPE, I labeled normal as 1 and abnormal as 2;
            # thus y_pred should be compatible and to this end I added 1 to each of its entries.
            y_pred = y_pred + 1

            print("Shapes: \n",
                  "y_pred", y_pred.shape, "\n",
                  "y_test", y_test.shape, "\n"
                  "y_pred:", y_pred, "\n" 
                  "y_test:", y_test
                  )

            results[repeat]["specifier"] = specifier
            results[repeat]["x_test"] = x_test
            results[repeat]["y_test"] = y_test
            results[repeat]["y_pred"] = y_pred

            # end of the program execution
            end = time.time()
            print("Execution time of repeat number " + repeat + " is:", end - start)

            results[repeat]["time"] = end - start

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

            run = util.save_model(run=run, model=model, name=alg_name, experiment_name=specifier)

            run.finish()

        with open("../results/"+specifier, "wb") as fp:
            pickle.dump(results, fp)

    else:
        print("Reproducing results")
        n_repeats = str(n_repeats)

        if "nn" in alg_name.lower():
            specifier = alg_name + ", loss=" + loss + ", opt=" + optimizer + \
                        ", repeat=" + n_repeats + ", pp=" + pp + " target_is_org=" + str(target_is_org)

        elif "rf" in alg_name.lower() or "gb" in alg_name.lower():
            specifier = alg_name + ", n_estimators=" + str(n_estimators) + \
                        ", repeat=" + n_repeats + ", pp=" + pp + " target_is_org=" + str(target_is_org)
        else:
            specifier = alg_name + ", repeat=" + n_repeats + ", pp=" + pp + \
                        " target_is_org=" + str(target_is_org)

        print("specifier:", specifier)

    util.print_the_evaluated_results(specifier, learning_method, )







