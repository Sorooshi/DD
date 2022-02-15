import argparse
import numpy as np
from collections import defaultdict

from dd_package.data.dyslexia_data import DyslexiaData
from dd_package.models.regression_estimators import RegressionEstimators


np.set_printoptions(suppress=True, precision=3, linewidth=140)


def args_parser(arguments):

    _pp = arguments.pp.lower()
    _tag = arguments.tag.lower()
    _run = arguments.run
    _data_name = arguments.data_name  # .lower()
    _note = arguments.note
    _loss = arguments.loss.lower()
    _alg_name = arguments.alg_name  # .lower()
    _group = arguments.group
    _project = arguments.project
    _n_units = arguments.n_units
    _n_epochs = arguments.n_epochs
    _optimizer = arguments.optimizer.lower()
    _batch_size = arguments.batch_size
    _learning_rate = arguments.learning_rate
    _n_estimators = arguments.n_estimators
    _output_dim = arguments.output_dim
    _n_clusters = arguments.n_clusters
    _n_repeats = arguments.n_repeats
    _target_is_org = arguments.target_is_org

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
        learning_rate, n_estimators, output_dim, n_clusters, \
        n_repeats, target_is_org = args_parser(arguments=args)

    dd = DyslexiaData(n_splits=2, n_repeats=2, )

    demos = dd.get_demo_datasets()
    ias = dd.get_ia_datasets()
    fixs = dd.get_fix_datasets()

    demo = dd.concat_classes_demo()
    ia = dd.concat_classes_ia()
    fix = dd.concat_classes_fix()

    fix_demo = dd.concat_dfs(df1=fix,
                             df2=demo,
                             features1=fix.columns,
                             features2=demo.columns[2:],
                             )

    ia_demo = dd.concat_dfs(df1=ia,
                            df2=demo,
                            features1=ia.columns,
                            features2=demo.columns[2:],
                            )

    learning_method = None

    if learning_method == "regression":
        re = RegressionEstimators()
        estimator, params = re.instantiate_an_estimator()
        tuned_parameters = re.tune_hyper_parameters(estimator=estimator, params=params)
        results = re.train_test_tuned_estimator(estimator=estimator, tuned_params=tuned_parameters)




