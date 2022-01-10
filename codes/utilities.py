import os
import glob
import wandb
import numpy as np
import pandas as pd
from sklearn import metrics
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.utils import resample
from scipy.spatial import distance
from sklearn.metrics import roc_auc_score
from scipy.stats import wasserstein_distance
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, RobustScaler


np.set_printoptions(suppress=True, precision=3)


def load_data(data_name, group):

    """
        :param data_name: string, name of dataset
        :param group: string, W&B group name, i.e., "
                "using Features: FClustering, FClassification, FRegression, "
                "or using Time-series: TClustering, TClassification, TRegression")
        :return:
            data_org := Pandas df, concatenated (normal + abnormal) not preprocessed dataset.
            x := numpy, independent variables (features)
            y := numpy dependent variables (target values)
            indicators := list, a list of subject id, sentence etc depending on the dataset structure

    """
    # load the corresponding data
    data = {}
    # demographic data
    if data_name == "dd_demo":

        demo_xls = pd.ExcelFile("../data/demo.xlsx")
        indicators = ['SubjID', ]

        data["normal"] = remove_missing_data(
            df=pd.read_excel(demo_xls, "norm").sort_values(by=indicators)
        )

        data["normal"].replace({"fem": 1, "f": 1, "masc": -1, "m": -1}, inplace=True)

        data["abnormal"] = remove_missing_data(
            df=pd.read_excel(demo_xls, "dyslexia").sort_values(by=indicators)
        )

        data["abnormal"].replace({"fem": 1, "f": 1, "masc": -1, "m": -1}, inplace=True)

        q_features = ['Age', 'IQ', 'Sound_detection', 'Sound_change', ]
        c_features = ['Sex', 'Grade', ]
        features = q_features + c_features

        if "regression" in group.lower():
            targets = ['Reading_speed', ]
        else:
            targets = ['Group', ]

    # ia_report data
    elif data_name == "dd_ia":

        ia_report_xls = pd.ExcelFile("../data/IA_report.xlsx")

        indicators = ['SubjectID', 'Sentence_ID', 'Word_Number', ]

        data["normal"] = remove_missing_data(
            df=pd.read_excel(ia_report_xls, "norm").sort_values(
                by=indicators,
                axis=0)
        )

        data["abnormal"] = remove_missing_data(
            df=pd.read_excel(ia_report_xls, "dyslexia").sort_values(
                by=indicators,
                axis=0)
        )

        q_features = ['FIXATION_COUNT', 'TOTAL_READING_TIME',
                      'FIRST_FIXATION_DURATION', 'FIRST_FIXATION_X',
                      'FIRST_FIXATION_Y', 'FIRST_RUN_TOTAL_READING_TIME',
                      'FIRST_SACCADE_AMPLITUDE', 'REGRESSION_IN', 'REGRESSION_OUT',
                      'REGRESSION_OUT_FULL', 'REGRESSION_PATH_DURATION']

        c_features = ['QUESTION_ACCURACY', 'SKIP', ]

        features = q_features + c_features

        if "regression" in group.lower():
            targets = ['Reading_speed', ]
        else:
            targets = ['Group', ]

    # fixation report data
    elif data_name == "dd_fix":

        fixation_xls = pd.ExcelFile("../data/Fixation_report.xlsx")
        indicators = ['SubjectID', 'Sentence_ID', 'Word_Number', ]

        data["normal"] = remove_missing_data(
            df=pd.read_excel(fixation_xls, "norm").sort_values(
                by=indicators,
                axis=0)
        )

        data["abnormal"] = remove_missing_data(
            df=pd.read_excel(fixation_xls, "dyslexia").sort_values(
                by=indicators,
                axis=0)
        )

        q_features = ['FIX_X', 'FIX_Y', 'FIX_DURATION', ]
        c_features = None

        features = q_features

        if "regression" in group.lower():
            targets = ['Reading_speed', ]
        else:
            targets = ['Group', ]
    else:
        print("Undefined data set, define it above!")

    data_org = pd.concat([data["normal"], data["abnormal"]])
    data_org.replace({"norm": 1, "dyslexia": -1}, inplace=True)

    if c_features:
        pd.get_dummies(data_org, columns=c_features)

    # I should modify here and after it
    x = data_org.loc[:, features].values
    y = data_org.loc[:, targets].values

    return data_org, x, y, features, targets, indicators


def remove_missing_data(df):
    for col in df.columns:
        try:
            df[col].replace({".": np.nan}, inplace=True)

        except Exception as e:
            print(e, "\n No missing values in", col)
    return df.dropna()


def mae(y_trues, y_preds):
    if not isinstance(y_trues, np.ndarray):
        y_trues = np.asarray(y_trues)

    if not isinstance(y_preds, np.ndarray):
        y_preds = np.asarray(y_preds)

    return np.mean(np.abs(y_trues-y_preds))


def rmse(y_trues, y_preds):
    if not isinstance(y_trues, np.ndarray):
        y_trues = np.asarray(y_trues)

    if not isinstance(y_preds, np.ndarray):
        y_preds = np.asarray(y_preds)

    return np.sqrt(np.mean(np.power(y_trues-y_preds, 2)))


def mrae(y_trues, y_preds):
    if not isinstance(y_trues, np.ndarray):
        y_trues = np.asarray(y_trues)

    if not isinstance(y_preds, np.ndarray):
        y_preds = np.asarray(y_preds)
    return np.mean(np.abs(np.divide(y_trues -y_preds, y_trues)))


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
    return errors.mean(axis=0), errors.std(axis=0)


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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=True)
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
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict_proba(X_test)[:, 1]
        auc = 2 * roc_auc_score(y_test, y_pred_test) - 1
        scores.append(auc)

    scores = np.array(scores)
    mean = scores.mean()
    std = scores.std() / np.sqrt(len(scores))

    return mean, std


def range_standardizer(x):
    """ Returns Range standardized data set.
    Input: a numpy array, representing entity-to-feature matrix.
    """

    x_rngs = np.ptp(x, axis=0)
    x_means = np.mean(x, axis=0)

    x_r = np.divide(np.subtract(x, x_means), x_rngs)  # range standardization

    return np.nan_to_num(x_r)


def range_standardizer_(x_test, x_train):
    """ Returns Range standardized data set.
    Input: a numpy array, representing entity-to-feature matrix.
    """

    x_rngs = np.ptp(x_train, axis=0)
    x_means = np.mean(x_train, axis=0)

    x_r = np.divide(np.subtract(x_test, x_means), x_rngs)  # range standardization

    return np.nan_to_num(x_r)


def zscore_standardizer(x):
    """ Returns Z-scored standardized data set.
    Input: a numpy array, representing entity-to-feature matrix.
    """

    x_stds = np.std(x, axis=0)
    x_means = np.mean(x, axis=0)

    x_z = np.divide(np.subtract(x, x_means), x_stds)  # z-scoring

    return np.nan_to_num(x_z)


def zscore_standardizer_(x_test, x_train):
    """ Returns Z-scored standardized data set.
    Input: a numpy array, representing entity-to-feature matrix.
    """

    x_stds = np.std(x_train, axis=0)
    x_means = np.mean(x_train, axis=0)

    x_z = np.divide(np.subtract(x_test, x_means), x_stds)  # z-scoring

    return np.nan_to_num(x_z)


def quantile_standardizer(x, out_dist):

    QT =  QuantileTransformer(output_distribution=out_dist,)
    x_q = QT.fit_transform(x)

    return x_q, QT


def quantile_standardizer_(QT, x,):

    x_q = QT.fit_transform(x)

    return x_q


def _minmax_standardizer(x):
    x_mm = MinMaxScaler().fit_transform(x)
    return x_mm


def minmax_standardizer(x):
    x_mm = np.divide(np.subtract(x, x.min(axis=0)),
                     (x.max(axis=0) - x.min(axis=0)))
    return np.nan_to_num(x_mm)


def minmax_standardizer_(x_test, x_train):
    x_mm = np.divide(np.subtract(x_test, x_train.min(axis=0)),
                     (x_train.max(axis=0) - x_train.min(axis=0)))
    return np.nan_to_num(x_mm)


def robust_standardizer(x):
    RS = RobustScaler()
    x_rs = RS.fit_transform(x)
    return x_rs, RS


def robust_standardizer_(RS, x):
    x_rs = RS.fit_transform(x)
    return x_rs


def data_splitter(x, validation=False):

    if not validation:
        all_idx = np.arange(len(x))
        train_size = int(0.9 * len(all_idx))
        train_idx = np.random.choice(a=all_idx, size=train_size, replace=False, )
        test_idx = list(set(all_idx).difference(train_idx))
        return train_idx, test_idx

    elif validation:
        all_idx = np.arange(len(x))
        train_size = int(0.7 * len(all_idx))
        train_idx = np.random.choice(a=all_idx, size=train_size, replace=False, )
        test_idx = list(set(all_idx).difference(train_idx))
        test_size = int(0.5 * len(test_idx))
        val_idx = np.random.choice(test_idx, size=test_size, replace=False)
        test_idx = list(set(test_idx).difference(val_idx))
        return train_idx, val_idx, test_idx


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


# Old set of metrics: the means of two predicted targets values are computed
def _wandb_metrics(run, y_trues, y_preds):
    wasser = []
    for i in range(y_trues.shape[1]):
        wasser += [wasserstein_distance(y_trues[:, i], y_preds[:, i])]
    wasser = np.asarray(wasser)
    jsd = np.asarray(distance.jensenshannon(y_trues, y_preds))
    run.log({
        "MAE": mae(y_trues=y_trues, y_preds=y_preds),
        "RMSE": rmse(y_trues=y_trues, y_preds=y_preds),
        "MRAE": mrae(y_trues=y_trues, y_preds=y_preds),
        "R^2-Score": metrics.r2_score(y_trues, y_preds),
        "JSD": jsd.mean(),
        "Wasser": wasser.mean(),
    })

    return run


def evaluate_a_x_test(y_trues, y_preds,):

    # MEAPE >> Mean Estimation Absolute Percentage Error
    meape_mu, meape_std = mean_estimation_absolute_percentage_error(y_trues, y_preds, n_iters=100)

    # gb >> Gradient Decent Boosting Classifier
    gb_mu, gb_std = discrepancy_score(y_trues, y_preds, model='GDBT', n_iters=10)

    # qda >> Quadratic Discriminant Analysis
    qda_mu, qda_std = discrepancy_score(y_trues, y_preds, model='QDA', n_iters=10)

    return meape_mu, gb_mu, qda_mu


def wandb_metrics(run, meape_iops_mu, meape_lat_mu,  gb_mu, qda_mu,):

    """
    meape >> Mean Estimation Absolute Percentage Error
    gb >> Gradient Decent Boosting Classifier
    qda >> Quadratic Discriminant Analysis
    """

    run.log({
        "IOPS: MEAPE [mu, std]": ["%.3f" % meape_iops_mu.mean(), "%.3f" % meape_iops_mu.std()],
        "LAT : MEAPE [mu, std]": ["%.3f" % meape_lat_mu.mean(), "%.3f" % meape_lat_mu.std()],
        "DS_GBDT: [mu, std]": ["%.3f" % gb_mu.mean(), "%.3f" % gb_mu.std()],
        "DS_QDA: [mu, std]": ["%.3f" % qda_mu.mean(), "%.3f" % qda_mu.std()],
    })

    return run


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


def wandb_plot_total_predictions(run, algorithm, y_trues, y_preds, run_no):

    n_outputs = y_trues.shape[1]
    for j in range(n_outputs):
        target_name = "iops"
        t = np.arange(len(y_trues[:, j]))
        fig, ax = plt.subplots(1, figsize=(15, 10))
        ax.plot(t, y_trues[:, j], lw=1.5, c='g', label="y_trues", alpha=0.5)
        ax.plot(t, y_preds[:, j], lw=2., c='m', label="y_preds", alpha=0.5)
        if j == 1: target_name = "lat"
        ax.fill_between(t, y_preds[:, j] - y_preds[:, j].std(),
                        y_preds[:, j] + y_preds[:, j].std(),
                        facecolor='blue', alpha=0.7,
                        label=algorithm + "-" +  target_name
                        )
        ax.legend(loc="best")

        # run.log({"True & Pred. values of " + algorithm + str(j+1) + "-th preds.":
        #              wandb.Image(ax, caption="std of " + algorithm)})

        r2 = metrics.r2_score(y_trues[:, j], y_preds[:, j])

        # plt.savefig("../figures/" + target_name +  " run_num=" + run_no + ".png" )

        run.log({algorithm + ": " + target_name + ", run_num=" + run_no + ": R^2-Score= %.3f" %r2 : ax})

    return run


def wandb_plot_per_feature_predictions(run, features, x_test_org, x_test, model, algorithm, y_trues):

    for i in range(len(features)):
        for j in range(i, len(features)):

            rows = list(set(
                np.where(x_test_org[:, i] == 1)[0].tolist()).intersection(np.where(x_test_org[:, j] == 1)[0].tolist())
                        )

            if len(rows) > 0:
                x_test_slice = x_test[rows, :]
                y_test_slice = y_trues[rows, :]
                y_preds_slice = model.predict(x_test_slice)

                # print("Average results for:" + features[i] + " and " + features[j], " \n",
                #       "\t MRAE: %.3f" % mrae(y_trues=y_test_slice, y_preds=y_preds_slice),
                #       "R^2-Scoer: %.3f" % metrics.r2_score(y_true=y_test_slice, y_pred=y_preds_slice),
                #       "\n"
                #       )

                mrae_0 = mrae(y_trues=y_test_slice[:, 0], y_preds=y_preds_slice[:, 0])
                r2_0 = metrics.r2_score(y_true=y_test_slice[:, 0], y_pred=y_preds_slice[:, 0])

                plt.figure(figsize=(12, 8))
                plt.plot(y_test_slice[:, 0], c="g", marker=".", alpha=.4, label="iops-y_true")
                plt.plot(y_preds_slice[:, 0], c="b", alpha=.6, label="iops-y_pred")
                plt.legend()
                plt.title("MRAE = %.3f" % mrae_0 + " R2Score = %.3f" % r2_0)
                run.log({ "IOPS: R^2 score= %.3f"  % r2_0 + " of " + features[i] + " & " + features[j] : plt})
                plt.show()

                mrae_1 = mrae(y_trues=y_test_slice[:, 1], y_preds=y_preds_slice[:, 1])
                r2_1 = metrics.r2_score(y_true=y_test_slice[:, 1], y_pred=y_preds_slice[:, 1])

                plt.figure(figsize=(12, 8))
                plt.plot(y_test_slice[:, 1], c="g", marker=".", alpha=.4, label="lat-y_true")
                plt.plot(y_preds_slice[:, 1], c="b", alpha=.6, label="lat-y_pred")
                plt.legend()
                plt.title("MRAE = %.3f" % mrae_1 + " R2Score = %.3f" % r2_1)
                run.log({ "LAT: R^2 score=  %.3f"  % r2_1 + " of " + features[i] + " & " + features[j]: plt})
                plt.show()

            else:
                print("no intersection for", i, j)

    return run


def basic_plots(run, y_true, y_pred, run_no, specifier, title="", ):

    iops_true = y_true[:, 0]
    iops_pred = y_pred[:, 0]

    lat_true = y_true[:, 1]   # / 10 ** 6  # bcs I don't inverse the predictions
    lat_pred = y_pred[:, 1]   # / 10 ** 6  # bcs I don't inverse the predictions

    plt.figure(figsize=(21, 6))

    plt.subplot(131)
    plt.scatter(iops_true, lat_true, alpha=0.5, marker='o', label='True')
    plt.scatter(iops_pred, lat_pred, alpha=0.5, marker='+', label='Prediction')
    plt.xlabel('IOPS', size=14)
    plt.ylabel('Latency, ms', size=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.title(title, size=14)
    plt.legend(loc='best', fontsize=10)
    # run.log({"Scatter of IOPS vs LAT run=" + run_no: plt})
    # plt.show()

    plt.subplot(132)
    vals = np.concatenate((iops_true, iops_pred))
    bins = np.linspace(vals.min(), vals.max(), 50)
    plt.hist(iops_true, bins=bins, alpha=1., label='True', histtype='step', linewidth=3)
    plt.hist(iops_pred, bins=bins, alpha=1., label='Prediction')
    plt.xlabel('IOPS', size=14)
    plt.ylabel('Counts', size=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.title(title, size=14)
    plt.legend(loc='best', fontsize=10)
    # run.log({"Hist. of IOPS run=" + run_no: plt})
    # plt.show()

    plt.subplot(133)
    vals = np.concatenate((lat_true, lat_pred))
    bins = np.linspace(vals.min(), vals.max(), 50)
    plt.hist(lat_true, bins=bins, alpha=1., label='True', histtype='step', linewidth=3)
    plt.hist(lat_pred, bins=bins, alpha=1., label='Prediction')
    plt.xlabel('Latency, ms', size=14)
    plt.ylabel('Counts', size=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.title(title, size=14)
    plt.legend(loc='best', fontsize=10)
    # plt.savefig("../figures/" + specifier + " num_run=" + run_no + " op-" + title + ".png")
    run.log({ specifier + "-num_run=" + run_no + " op-" + title:
                  wandb.Image(plt, caption=specifier + " num_run=" + run_no + " op-" + title)})
    plt.show()


def plot_true_pred_distributions(run, y_test, y_pred, algorithm, n_bins):


    for i in range(y_test.shape[1]):
        _ = plt.figure(figsize=(12, 8))
        plt.hist(y_test[:, i], color="g", bins=n_bins, label="y_true", histtype='step')
        plt.hist(y_pred[:, i], color="b", bins=n_bins, label="y_pred", histtype='step')
        _max = max(y_test[:, i].max(), y_pred[:, i].max()) + 400
        plt.xlim([0, _max])
        plt.xlabel("True and Pred. values' Distributions " )
        plt.ylabel('Count')
        plt.legend(loc="best")
        run.log({"Distributions of " + algorithm + str(i+1) + "-th preds.": plt})

    return run


def plot_loss(run, history, name):

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.plot(history.history['loss'], label='Train Loss-' + name)
    ax.plot(history.history['val_loss'], label='Valid. Loss-' + name)
    plt.ylabel("Error")
    plt.xlabel("Epochs")
    plt.title("Train-Validation Errors for " + name)
    ax.legend(loc="best")
    run.log({"Train-Validation Errors for" + name: ax})
    # plt.savefig("../figures/"+name+".png")
    # plt.show()

    return run


def plot_predictions(y_test, y_pred, name):
    _ = plt.figure(figsize=(13.5, 7.5))
    plt.scatter(y_test, y_pred)
    plt.xlabel("True Values (" + name + ")")
    plt.ylabel("True Values (" + name + ")")
    plt.title("Scatter plot of target values vs predicted values")


def plot_error_distribution(y_test, y_pred, name, n_bins):
    error = y_pred - y_test
    plt.hist(error, bins=n_bins)
    plt.xlabel("Prediction Error (" + name + ")")
    plt.ylabel('Count')
    return None


def display_samples(x, y=None):
    if not isinstance(x, (np.ndarray, np.generic)):
        x = np.array(x)
    n = x.shape[0]
    fig, axs = plt.subplots(1, n, figsize=(n, 1))
    if y is not None:
        fig.suptitle(np.argmax(y, axis=1))
    for i in range(n):
        axs.flat[i].plot(x[i].squeeze(), )
        axs.flat[i].axis('off')
    plt.show()
    plt.close()


def save_model(run, model, name, experiment_name):

    if name == "rfr" or name == "gbr" or name == "gmm" or name == 'ar':
        dump(model, os.path.join(wandb.run.dir, name + experiment_name + ".joblib"))  #

    elif name == "dnn_reg" or name=="vnn_reg":
        model.save_weights(os.path.join(wandb.run.dir,
                                        name + experiment_name + ".h5"))
    # else:
    #     model.save(os.path.join(wandb.run.dir,
    #                             name + experiment_name + ".h5"))

    return run



