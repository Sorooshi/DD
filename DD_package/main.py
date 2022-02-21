import argparse
import numpy as np
from pathlib import Path
from types import SimpleNamespace

from dd_package.models.baseline import BaseLineModel
from dd_package.data.preprocess import preprocess_data
from dd_package.data.dyslexia_data import DyslexiaData
from dd_package.models.regression_estimators import RegressionEstimators
from dd_package.models.classification_estimators import ClassificationEstimators


np.set_printoptions(suppress=True, precision=3, linewidth=140)


def args_parser(arguments):

    _pp = arguments.pp.lower()
    _tag = arguments.tag.lower()
    _run = arguments.run
    _data_name = arguments.data_name.lower()
    _estimator_name = arguments.estimator_name.lower()
    _project = arguments.project
    _target_is_org = arguments.target_is_org
    _to_shuffle = arguments.to_shuffle
    _n_clusters = arguments.n_clusters
    _to_exclude_at_risk = arguments.to_exclude_at_risk

    return _pp, _tag, _run, _data_name, _estimator_name, _project,\
           _target_is_org, _to_shuffle, _n_clusters, _to_exclude_at_risk


configs = {
    "models_path": Path("/home/soroosh/Programmes/DD/Models"),
    "results_path": Path("/home/soroosh/Programmes/DD/Results"),
    "figures_path": Path("/home/soroosh/Programmes/DD/Figures"),
    "params_path": Path("/home/soroosh/Programmes/DD//Params"),
    "n_repeats": 10,
    "n_splits": 5,
}

configs = SimpleNamespace(**configs)

if not configs.models_path.exists():
    configs.models_path.mkdir()

if not configs.results_path.exists():
    configs.results_path.mkdir()

if not configs.figures_path.exists():
    configs.figures_path.mkdir()

if not configs.params_path.exists():
    configs.params_path.mkdir()


if __name__ == "__main__":

    # all the string inputs will be converted to lower case.
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project", type=str, default="DD",
        help="Project name for WandB project initialization."
    )

    parser.add_argument(
        "--data_name", type=str, default="DD_Demo",
        help="Dataset's name, e.g., DD_Demo, or DD_Demo_IA."
             "The following (lowercase) strings are supported"
             "  1) Demographic = dd_demo, "
             "  2) IA_report = dd_ia, "
             "  3) Fixation_report = dd_fix, "
             "  4) Demographic + IA_report = dd_demo_ia, "
             "  5) Demographic + Fixation_report = dd_demo_fix,"
             "  6) IA_report + demo.Reading_speed = dd_ia_reg,"
             "  7) Fix_report + demo.Reading_speed = dd_fix_reg"
    )

    parser.add_argument(
        "--estimator_name", type=str, default="base_reg",
        help="None case sensitive first letter abbreviated name of an estimator proceeds "
             "  one of the three following suffixes separated with the underscore."
             "  Possible suffixes are: regression := reg, "
             "  classification := cls, clustering := clu"
             "      E.g., Random Forest Regressor := rf_reg, or "
             "      Random Forest Classifiers := rf_cls "
             "Note: First letter of the methods' name should be used for abbreviation."
    )

    parser.add_argument(
        "--run", type=int, default=1,
        help="Run the model or load the saved"
             " weights and reproduce the results."
    )

    parser.add_argument(
        "--pp", type=str, default="mm",
        help="Data preprocessing method:"
             " MinMax/Z-Scoring/etc."
    )

    parser.add_argument(
        "--tag", type=str, default="warmup",
        help="W&B tag will be used to filter some of runs"
             "of the same set of experiments if needed."
    )

    parser.add_argument(
        "--note", type=str, default="--",
        help="W&B note, e.g., clustering for DD: Demographic"
    )

    parser.add_argument(
        "--n_clusters", type=int, default=3,
        help="Number of clusters/classes/discrete target values."
    )

    parser.add_argument(
        "--target_is_org", type=int, default=1,
        help="Whether to use not preprocessed target values or not."
    )

    parser.add_argument(
        "--to_shuffle", type=int, default=1,
        help="Whether to shuffle data during CV or not."
             "  Only setting it to one (shuffle=1) will shuffle data."
    )

    parser.add_argument(
        "--to_exclude_at_risk", type=int, default=0,
        help="Whether to exclude at-risk class from experiments or not."
             "  Only setting it to one (to_exclude_at_risk=1) will exclude this class. "
    )

    args = parser.parse_args()

    pp, tag, run, data_name, estimator_name, project, \
        target_is_org, to_shuffle, n_clusters, to_exclude_at_risk = args_parser(arguments=args)

    print(
        "configuration: \n",
        "  estimator:", estimator_name, "\n",
        "  data_name:", data_name, "\n",
        "  shuffle_data:", to_shuffle, "\n",
        "  pre-processing:", pp, "\n",
        "  run:", run, "\n",
        "  to_exclude_at-risk:", to_exclude_at_risk, "\n",
    )

    dd = DyslexiaData(
        n_splits=configs.n_splits,
        n_repeats=configs.n_repeats,
    )

    # dict of dicts, s.t each dict contains pd.df of a class, e.g normal
    _ = dd.get_demo_datasets()  # demos
    _ = dd.get_ia_datasets()  # ias
    _ = dd.get_fix_datasets()  # fixes

    # The three below lines can be move to
    #   if data_names == ... for less memory consumption, if it needed.

    # concatenate pd.dfs to a pd.df
    ia = dd.concat_classes_ia()
    fix = dd.concat_classes_fix()
    demo = dd.concat_classes_demo()

    # The optimize way to exclude at-risk class
    if to_exclude_at_risk == 1:
        to_exclude_at_risk = True
        ia = ia.loc[ia.Group != 2]
        fix = fix.loc[fix.Group != 2]
        demo = demo.loc[demo.Group != 2]

    # Determine which dataset to use, e.g. demo dataset
    # alone or concatenation of demo and IA_report, for instance.
    if data_name == "dd_demo":
        df_data_to_use = demo
        c_features = ['Sex', 'Grade', ]
        indicators = ['SubjectID', ]
        targets = ["Group", "Reading_speed", ]

    elif data_name == "dd_ia":
        df_data_to_use = ia
        c_features = [
            'QUESTION_ACCURACY', 'SKIP', 'REGRESSION_IN',
            'REGRESSION_OUT', 'REGRESSION_OUT_FULL',
        ]

        indicators = [
            'SubjectID', 'Sentence_ID', 'Word_Number',
        ]

        targets = ["Group", ]

    elif data_name == "dd_fix":
        df_data_to_use = fix
        c_features = None
        indicators = [
            'SubjectID', 'Sentence_ID', 'Word_Number',
        ]

        targets = ["Group", ]

    elif data_name == "dd_ia_demo":

        ia_demo = dd.concat_dfs(
            df1=ia,
            df2=demo,
            features1=ia.columns,
            features2=demo.columns[2:],
        )

        df_data_to_use = ia_demo
        c_features = [
            'Sex', 'Grade', 'QUESTION_ACCURACY',
            'SKIP', 'REGRESSION_IN', 'REGRESSION_OUT',
            'REGRESSION_OUT_FULL',
        ]

        indicators = [
            'SubjectID', 'Sentence_ID', 'Word_Number',
        ]

        targets = ["Group", "Reading_speed", ]

    elif data_name == "dd_ia_reg":

        ia_reg = dd.concat_dfs(
            df1=ia,
            df2=demo,
            features1=ia.columns,
            features2=["Reading_speed", ],
        )

        df_data_to_use = ia_reg

        c_features = [
            'QUESTION_ACCURACY', 'SKIP', 'REGRESSION_IN',
            'REGRESSION_OUT', 'REGRESSION_OUT_FULL',
        ]

        indicators = [
            'SubjectID', 'Sentence_ID', 'Word_Number',
        ]

        targets = ["Group", "Reading_speed", ]

    elif data_name == "dd_fix_demo":

        fix_demo = dd.concat_dfs(
            df1=fix,
            df2=demo,
            features1=fix.columns,
            features2=demo.columns[2:],
        )

        df_data_to_use = fix_demo
        c_features = ['Sex', 'Grade', ]
        indicators = [
            'SubjectID', 'Sentence_ID', 'Word_Number',
        ]

        targets = ["Group", "Reading_speed", ]

    elif data_name == "dd_fix_reg":

        fix_reg = dd.concat_dfs(
            df1=fix,
            df2=demo,
            features1=fix.columns,
            features2=["Reading_speed"],
        )

        df_data_to_use = fix_reg
        c_features = None
        indicators = [
            'SubjectID', 'Sentence_ID', 'Word_Number',
        ]

        targets = ["Group", "Reading_speed", ]

    else:
        print("data_name argument:", data_name)
        assert False, "Ill-defined data_name argument. Refer to help of data_name argument for more."

    x_org, y_org = dd.get_onehot_features_targets(
        data=df_data_to_use,
        c_features=c_features,
        indicators=indicators,
        targets=targets,
    )

    print("x_org:", x_org.shape, "\n", x_org.head())

    if estimator_name.split("_")[-1] == "reg":
        learning_method = "regression"
        y = y_org.Reading_speed.values

    elif estimator_name.split("_")[-1] == "cls":
        learning_method = "classification"
        y = y_org.Group.values

    elif estimator_name.split("_")[-1] == "clu":
        learning_method = "clustering"
        y = y_org.Group.values
    else:
        assert False, "Undefined algorithm and thus undefined target values"

    if to_shuffle == 1:
        to_shuffle = True
        group = learning_method + "-" + "shuffled"
    else:
        to_shuffle = False
        group = learning_method + "-" + "not-shuffled"

    # Adding some details for the sake of clarity in storing and visualization
    configs.run = run
    configs.project = project
    configs.group = group
    configs.tag = tag
    specifier = data_name + "-" + estimator_name + \
                "--shuffled:" + str(to_shuffle) + \
                "--exclude at risk:" + str(to_exclude_at_risk)
    configs.specifier = specifier
    configs.data_name = data_name
    configs.name_wb = data_name+": "+specifier
    configs.learning_method = learning_method

    x = preprocess_data(x=x_org, pp=pp)  # only x is standardized

    cv = dd.get_stratified_kfold_cv(
        to_shuffle=to_shuffle,
        n_splits=configs.n_splits
    )

    data = dd.get_stratified_train_test_splits(
        x=x, y=y,
        labels=y_org.Group.values,
        to_shuffle=to_shuffle,
        n_splits=configs.n_repeats
    )

    # Baseline models (random prediction)
    if estimator_name == "base_reg" or \
            estimator_name == "base_cls" or \
            estimator_name == "base_clu":
        blm = BaseLineModel(
            y_train=y,
            learning_method=learning_method,
            configs=configs,
            test_size=200
        )

        blm.repeat_random_pred()
        blm.save_results()
        blm.print_results()

        assert False, "Random prediction is done, no need to proceed further"

    # Regression methods:
    if learning_method == "regression":

        reg_est = RegressionEstimators(
            x=x, y=y, cv=cv, data=data,
            estimator_name=estimator_name,
            configs=configs,
        )

        reg_est.instantiate_tuning_estimator_and_parameters()

        reg_est.tune_hyper_parameters()

        reg_est.instantiate_train_test_estimator()

        reg_est.train_test_tuned_estimator()

        reg_est.save_params_results()

        reg_est.print_results()

    # Classification methods:
    if learning_method == "classification":
        print(
            "classification to be completed"
        )

        cls_est = ClassificationEstimators(
            x=x, y=y, cv=cv, data=data,
            estimator_name=estimator_name,
            configs=configs,
        )

        cls_est.instantiate_tuning_estimator_and_parameters()

        cls_est.tune_hyper_parameters()

        cls_est.instantiate_train_test_estimator()

        cls_est.train_test_tuned_estimator()

        cls_est.save_params_results()

        cls_est.print_results()

    if learning_method == "clustering":
        print(
            "clustering to be completed"
        )

    print(
        "\n Hyper-parameters tuning and train-test evaluation at " + data_name + " are finished. \n",
        "  The corresponding results, parameters, models, and figures of " + estimator_name + " are stored."
    )


