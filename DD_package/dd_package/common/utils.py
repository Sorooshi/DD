import os
import wandb
import pickle
import numpy as np
from pathlib import Path
from sklearn import metrics
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.utils import resample
from scipy.spatial import distance
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier



np.set_printoptions(suppress=True, precision=3)


def save_a_dict(a_dict, name, save_path, ):
    with open(os.path.join(save_path, name+".pickle"), "wb") as fp:
        pickle.dump(a_dict, fp)
    return None


def load_a_dict(name, save_path, ):
    with open(os.path.join(save_path, name + ".pickle"), "rb") as fp:
        a_dict = pickle.load(fp)
    return a_dict


def mae(y_true, y_pred):
    if not isinstance(y_true, np.ndarray):
        y_true = np.asarray(y_true)

    if not isinstance(y_pred, np.ndarray):
        y_pred = np.asarray(y_pred)

    return np.mean(np.abs(y_true-y_pred))


def rmse(y_true, y_pred):
    if not isinstance(y_true, np.ndarray):
        y_true = np.asarray(y_true)

    if not isinstance(y_pred, np.ndarray):
        y_pred = np.asarray(y_pred)

    return np.sqrt(np.mean(np.power(y_true-y_pred, 2)))


def mrae(y_true, y_pred):
    if not isinstance(y_true, np.ndarray):
        y_true = np.asarray(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.asarray(y_pred)
    return np.mean(np.abs(np.divide(y_true - y_pred, y_true)))


def jsd(y_true, y_pred):
    return np.asarray(distance.jensenshannon(y_true, y_pred))


def mean_estimation_absolute_percentage_error(y_true, y_pred, n_iters=100):
    errors = []
    inds = np.arange(len(y_true))

    for i in range(n_iters):
        inds_boot = resample(inds)

        y_true_boot = y_true[inds_boot]
        y_pred_boot = y_pred[inds_boot]

        y_true_mean = y_true_boot.mean(axis=0)
        y_pred_mean = y_pred_boot.mean(axis=0)

        ierr = np.abs((y_true_mean - y_pred_mean) / y_true_mean) * 100
        errors.append(ierr)

    errors = np.array(errors)
    return errors


def discrepancy_score(observations, forecasts, model='QDA', n_iters=1):

    """
    Parameters:
    -----------
    observations : numpy.ndarray, shape=(n_samples, n_features)
        True values.
        Example: [[1, 2], [3, 4], [4, 5], ...]
    forecasts : numpy.ndarray, shape=(n_samples, n_features)
        Predicted values.
        Example: [[1, 2], [3, 4], [4, 5], ...]
    model : sklearn binary classifier
        Possible values: RF, DT, LR, QDA, GBDT
    n_iters : int
        Number of iteration per one forecast.

    Returns:
    --------
    mean : float
        Mean value of discrepancy score.
    std : float
        Standard deviation of the mean discrepancy score.

    """

    scores = []

    X0 = observations
    y0 = np.zeros(len(observations))

    X1 = forecasts
    y1 = np.ones(len(forecasts))

    X = np.concatenate((X0, X1), axis=0)
    y = np.concatenate((y0, y1), axis=0)

    for it in range(n_iters):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=True)
        if model == 'RF':
            clf = RandomForestClassifier(n_estimators=100, max_depth=10, max_features=None)
        elif model == 'GDBT':
            clf = GradientBoostingClassifier(max_depth=6, subsample=0.7)
        elif model == 'DT':
            clf = DecisionTreeClassifier(max_depth=10)
        elif model == 'LR':
            clf = LogisticRegression()
        elif model == 'QDA':
            clf = QuadraticDiscriminantAnalysis()
        clf.fit(x_train, y_train)
        y_pred_test = clf.predict_proba(x_test)[:, 1]
        auc = 2 * metrics.roc_auc_score(y_test, y_pred_test) - 1
        scores.append(auc)

    scores = np.array(scores)
    mean = scores.mean()
    std = scores.std() / np.sqrt(len(scores))

    return mean, std


def init_a_wandb(name, project, notes, group, tag, config):

    """ name := the within the project name, e.g., RF-reg-1
        project := the project name, e.g., Non-sequential Regressions
        notes := Description, e.g., Non-sequential Regressions Comparison for SuperOX
        group := name of experiment or the algorithm under consideration, e.g., RF-1
        config := model and training configuration
        tag := tag of an experiment, e.g. run number of same experiments to compute ave. and std.
    """

    run = wandb.init(name=name,
                     project=project,
                     notes=notes,
                     entity='sorooshi',
                     group=group,
                     tags=tag,
                     config=config,
                     )

    return run


def wandb_metrics(run, y_true, y_pred, y_pred_prob, learning_method):

    meape_errors = mean_estimation_absolute_percentage_error(y_true, y_pred, n_iters=100)

    # to compute ROC_AUC
    try:
        y_true.shape[1]
        y_true_ = y_true
    except:
        enc = OneHotEncoder(sparse=False)
        y_true_ = y_true.reshape(-1, 1)
        y_true_ = enc.fit_transform(y_true_)

    if learning_method == "regression":

        run.log({
            "MAE": mae(y_true=y_true, y_pred=y_pred),
            "RMSE": rmse(y_true=y_true, y_pred=y_pred),
            "MRAE": mrae(y_true=y_true, y_pred=y_pred),
            "JSD": jsd(y_true=y_true, y_pred=y_pred).mean(),
            "R^2-Score": metrics.r2_score(y_true, y_pred),
            "MEAPE-mu": meape_errors.mean(axis=0),
            "MEAPE-std": meape_errors.std(axis=0)
        })

    elif learning_method == "classification":

        run.log({
            "ARI": metrics.adjusted_rand_score(y_true, y_pred),
            "NMI": metrics.normalized_mutual_info_score(y_true, y_pred),
            "JSD": jsd(y_true=y_true, y_pred=y_pred).mean(),
            "Precision": metrics.precision_score(y_true, y_pred, average='weighted'),
            "Recall": metrics.recall_score(y_true, y_pred, average='weighted'),
            "F1-Score": metrics.f1_score(y_true, y_pred, average='weighted'),
            "ROC AUC": metrics.roc_auc_score(y_true_, y_pred_prob, average='weighted', multi_class="ovr"),
            "MEAPE-mu": meape_errors.mean(axis=0),
            "MEAPE-std": meape_errors.std(axis=0)
        })

    # for future applications I separate cls and clu
    elif learning_method == "clustering":

        run.log({
            "ARI": metrics.adjusted_rand_score(y_true, y_pred),
            "NMI": metrics.normalized_mutual_info_score(y_true, y_pred),
            "JSD": jsd(y_true=y_true, y_pred=y_pred).mean(),
            "Precision": metrics.precision_score(y_true, y_pred, average='weighted'),
            "Recall": metrics.recall_score(y_true, y_pred, average='weighted'),
            "F1-Score": metrics.f1_score(y_true, y_pred, average='weighted'),
            "ROC AUC": metrics.roc_auc_score(y_true_, y_pred_prob, average='weighted', multi_class="ovr"),
            "MEAPE-mu": meape_errors.mean(axis=0),
            "MEAPE-std": meape_errors.std(axis=0)
        })

    return run


def evaluate_a_x_test(y_true, y_pred,):

    # MEAPE >> Mean Estimation Absolute Percentage Error
    meape_mu, meape_std = mean_estimation_absolute_percentage_error(y_true, y_pred, n_iters=100)

    # gb >> Gradient Decent Boosting Classifier
    gb_mu, gb_std = discrepancy_score(y_true, y_pred, model='GDBT', n_iters=10)

    # qda >> Quadratic Discriminant Analysis
    qda_mu, qda_std = discrepancy_score(y_true, y_pred, model='QDA', n_iters=10)

    return meape_mu, gb_mu, qda_mu


def wandb_features_importance(run, values_features_importance,
                              name_important_features,
                              indices_important_features,
                              importance_method):
    counter = 0
    for i in range(len(indices_important_features)):
        if counter < 5:
            run.log({
                importance_method +
                "-" + name_important_features[indices_important_features[i]] +
                "-" + str(i + 1): values_features_importance[0][indices_important_features[i]],
            })
            counter += 1

    return run


def wandb_true_pred_plots(run, y_true, y_pred, path, specifier):

    t = np.arange(len(y_true))
    fig, ax = plt.subplots(1, figsize=(14, 7))
    ax.plot(t, y_true, lw=1.5, c='g', label="y_true", alpha=1.)
    ax.plot(t, y_pred, lw=2., c='m', label="y_pred", alpha=1.)

    ax.fill_between(t, y_pred + y_pred.std(),
                    y_pred - y_pred.std(),
                    facecolor='yellow',
                    alpha=.5,
                    label="Std",
                    )

    ax.legend(loc="best")
    r2 = metrics.r2_score(y_true, y_pred)

    plt.xlabel("Index")
    plt.ylabel("True/Pred Values")
    plt.legend(loc="best")

    plt.title(
        "Plots: target vs predicted value of " + specifier
    )

    # subdirectory
    p = Path(os.path.join(path, "Plots"))
    if not p.exists():
        p.mkdir()

    plt.savefig(
        os.path.join(
            p, specifier + ".png"
        )
    )

    run.log(
        {"Plots: target vs predicted value of " + specifier + str(r2): ax}
    )

    return run


def wandb_true_pred_scatters(run, y_test, y_pred, path, specifier,):

    _ = plt.figure(figsize=(14, 7))

    plt.scatter(np.arange(len(y_test)), y_test,
                alpha=0.7, marker='+', label='True')

    plt.scatter(np.arange(len(y_pred)), y_pred,
                alpha=0.8, marker='o', label='Prediction')

    plt.xlabel("Index")
    plt.ylabel("True/Pred Values ")
    plt.legend(loc="best")

    plt.title(
        "Scatters: target vs predicted values of " + specifier
    )

    # subdirectory
    p = Path(os.path.join(path, "Scatters"))
    if not p.exists():
        p.mkdir()

    plt.savefig(
        os.path.join(p, specifier + ".png")
    )

    run.log(
        {"Scatters: target vs predicted values of " + specifier: plt}
    )

    return run


def wandb_true_pred_histograms(run, y_test, y_pred, path, specifier):

    plt.figure(figsize=(15, 7))
    plt.subplot(131)
    n_bins = np.linspace(y_test.min()-5, y_test.max()+5, 50)

    plt.hist(y_test, color="g",
             bins=n_bins, label="y_true",
             histtype='step', alpha=.7,
             linewidth=2,
             )

    # n_bins = np.linspace(y_pred.min()-20, y_pred.max()+20, 50)

    plt.hist(y_pred, color="m",
             bins=n_bins, label="y_pred",
             histtype='step',
             alpha=1.,
             )

    _max = max(y_test.max(), y_pred.max()) + 20

    plt.xlim([-_max, _max])
    plt.xlabel("True and Pred. values ")
    plt.ylabel('Count')
    plt.legend(loc="best")

    plt.title(
        "Histograms: " + specifier
    )

    # subdirectory
    p = Path(os.path.join(path, "Histograms"))

    if not p.exists():
        p.mkdir()

    plt.savefig(
        os.path.join(
            p, specifier + ".png"
        )
    )

    run.log(
        {"Histograms: target vs predicted of " + specifier : plt}
    )

    plt.show()

    return run


def save_model(path, model, specifier, ):

    dump(
        model, os.path.join(
            path, specifier+".joblib"
        )
    )

    return None


def print_the_evaluated_results(results, learning_method, ):

    """ results: dict, containing results of each repeat, key:= repeat number.
            learning_method: string, specifing which metrics should be used.
    """

    # Regression metrics
    MEA, RMSE, MRAE, JSD, R2_Score, MEAPE_mu, MEAPE_std = [], [], [], [], [], [], []
    # Classification and clustering metrics
    ARI, NMI, Precision, Recall, F1_Score, ROC_AUC, ACC = [], [], [], [], [], [], []

    for repeat, result in results.items():
        y_true = result["y_test"]
        y_pred = result["y_pred"]
        y_pred_prob = result["y_pred_prob"]

        # to compute ROC_AUC
        try:
            y_true.shape[1]
            y_true_ = y_true
        except:
            enc = OneHotEncoder(sparse=False)
            y_true_ = y_true.reshape(-1, 1)
            y_true_ = enc.fit_transform(y_true_)

    if learning_method == "regression":

        MEA.append(mae(y_true=y_true, y_pred=y_pred))
        RMSE.append(rmse(y_true=y_true, y_pred=y_pred))
        MRAE.append(mrae(y_true=y_true, y_pred=y_pred))
        JSD.append(jsd(y_true=y_true, y_pred=y_pred).mean())
        R2_Score.append(metrics.r2_score(y_true, y_pred))
        meape_errors = mean_estimation_absolute_percentage_error(
            y_true=y_true, y_pred=y_pred, n_iters=100
        )
        MEAPE_mu.append(meape_errors.mean(axis=0))
        MEAPE_std.append(meape_errors.std(axis=0))

    else:
        ARI.append(metrics.adjusted_rand_score(y_true, y_pred))
        NMI.append(metrics.normalized_mutual_info_score(y_true, y_pred))
        JSD.append(jsd(y_true=y_true, y_pred=y_pred).mean())
        Precision.append(metrics.precision_score(y_true, y_pred, average='weighted'))
        Recall.append(metrics.recall_score(y_true, y_pred, average='weighted'))
        F1_Score.append(metrics.f1_score(y_true, y_pred, average='weighted'))
        ROC_AUC.append(metrics.roc_auc_score(y_true_, y_pred_prob, average='weighted', multi_class="ovr"),)
        meape_errors = mean_estimation_absolute_percentage_error(
                y_true=y_true, y_pred=y_pred, n_iters=100
        )

        MEAPE_mu.append(meape_errors.mean(axis=0))
        MEAPE_std.append(meape_errors.std(axis=0))
        ACC.append(metrics.accuracy_score(y_true, y_pred, ))

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

        print("   mae ", "   rmse ", "\t mrae",
              "\t r2_score ", "\t meape ", "\t jsd ",
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
        ACC = np.nan_to_num((np.asarray(ACC)))

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

        acc_ave = np.mean(ACC, axis=0)
        acc_std = np.std(ACC, axis=0)

        print("  ari ", "  nmi ", "\t preci", "\t recall ",
              "\t f1_score ", "\t roc_auc ", "\t meape ", "\t jsd ", "\t acc"
              )

        print(" Ave ", " std", " Ave ", " std ", " Ave ", " std ", " Ave ", " std ",
              " Ave ", " std ", " Ave ", " std ", " Ave ", " std ", " Ave ", " std ", " Ave ", " std "
              )

        print("%.3f" % ari_ave, "%.3f" % ari_std,
              "%.3f" % nmi_ave, "%.3f" % nmi_std,
              "%.3f" % precision_ave, "%.3f" % precision_std,
              "%.3f" % recall_ave, "%.3f" % recall_std,
              "%.3f" % f1_score_ave, "%.3f" % f1_score_std,
              "%.3f" % roc_auc_ave, "%.3f" % roc_auv_std,
              "%.3f" % meape_ave, "%.3f" % meape_std,
              "%.3f" % jsd_ave, "%.3f" % jsd_std,
              "%.3f" % acc_ave, "%.3f" % acc_std,
              )

    return None





