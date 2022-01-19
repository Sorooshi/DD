import numpy as np
import pandas as pd
import utilities as util
from sklearn.preprocessing import MinMaxScaler, \
    StandardScaler, QuantileTransformer, RobustScaler


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
    data_name = data_name.lower()

    if data_name == "dd_demo":

        demo_xls = pd.ExcelFile("../data/demo.xlsx")
        indicators = ['SubjID', ]

        data["normal"] = remove_missing_data(
            df=pd.read_excel(demo_xls, "norm").sort_values(by=indicators)
        )

        data["normal"].replace({"fem": 1, "f": 1, "masc": 2, "m": 2}, inplace=True)

        data["abnormal"] = remove_missing_data(
            df=pd.read_excel(demo_xls, "dyslexia").sort_values(by=indicators)
        )

        data["abnormal"].replace({"fem": 1, "f": 1, "masc": 2, "m": 2}, inplace=True)

        q_features = ['Age', 'IQ', 'Sound_detection', 'Sound_change', ]
        c_features = ['Sex', 'Grade', ]

        if "regression" in group.lower():  # or "baseline" in group.lower()
            targets = ['Reading_speed', ]
        else:
            targets = ['Group', ]

        all_targets = ['Reading_speed', 'Group']

    # ia_report data
    elif data_name == "dd_ia":
        # print("Load IA report!")

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

        # if "regression" in group.lower():
        #     targets = ['Reading_speed', ]
        # else:
        #     targets = ['Group', ]

        targets = ['Group', ]
        all_targets = ['Group', ]

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
        c_features = []

        if "regression" in group.lower():  # or "baseline" in group.lower():  # for rnd evaluation
            targets = ['Reading_speed', ]
        else:
            targets = ['Group', ]

        all_targets = ['Reading_speed', 'Group']

    else:
        print("Undefined data set, define it above!")

    _data_org = pd.concat([data["normal"], data["abnormal"]])
    _data_org = _data_org.replace({"norm": 1, "dyslexia": 2}, )

    if len(c_features) > 0:
        data_org = pd.get_dummies(_data_org, columns=c_features)
    else:
        data_org = _data_org

    all_features = data_org.columns
    features = set(all_features) - set(indicators) - set(all_targets)

    # I should modify here and after it
    x = data_org.loc[:, features].values
    y = data_org.loc[:, targets].values
    
    x = x.astype(float)
    # y = y.astype(float)

    print("Check data for NaNs or Inf: \n",
          "x: ",  np.where(x == np.inf), np.where(x == np.nan), "\n",
          "y: ", np.where(y == np.inf), np.where(y == np.nan), "\n",
          "shapes:", x.shape, y.shape,
          )

    return data_org, x, y, features, targets, indicators


def remove_missing_data(df):
    for col in df.columns:
        try:
            df[col].replace({".": np.nan}, inplace=True)
        except Exception as e:
            print(e, "\n No missing values in", col)

    return df.dropna()


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

    QT = QuantileTransformer(output_distribution=out_dist,)
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


def preprocess_data(x, y, pp):

    if pp == "rng":
        print("pre-processing:", pp)
        x = range_standardizer(x=x)
        y = range_standardizer(x=y)
        print("Preprocessed x and y shapes:", x.shape, y.shape)
    elif pp == "zsc":
        print("pre-processing:", pp)
        x = zscore_standardizer(x=x)
        y = zscore_standardizer(x=y)
        print("Preprocessed x and y shapes:", x.shape, y.shape)
    elif pp == "mm":  # MinMax
        print("pre-processing:", pp)
        x = minmax_standardizer(x=x)
        y = minmax_standardizer(x=y)
    elif pp == "rs":  # Robust Scaler (subtract median and divide with [q1, q3])
        print("pre-processing:", pp)
        x, rs_x = robust_standardizer(x=x)
        y, rs_y = robust_standardizer(x=y)
    elif pp == "qtn":  # quantile_transformation with Gaussian distribution as output
        x, qt_x = quantile_standardizer(x=x, out_dist="normal")
        y, qt_y = quantile_standardizer(x=y, out_dist="normal")
    elif pp == "qtu":  # quantile_transformation with Uniform distribution as output
        x, qt_x = quantile_standardizer(x=x, out_dist="uniform")
        y, qt_y = quantile_standardizer(x=y, out_dist="uniform")
    elif pp is None:
        x_org = x
        y_org = y
        print("No pre-processing")
    else:
        print("Undefined pre-processing")

    return x, y


def data_index_splitter(x, validation=False,):

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


def data_splitter(x, y, x_org, y_org, target_is_org):

    # train, validation and test split:
    train_idx, val_idx, test_idx = data_index_splitter(x=x, validation=True)

    x_train, y_train = x[train_idx, :], y[train_idx, :].ravel()
    x_val, y_val = x[val_idx, :], y[val_idx, :].ravel()
    x_test, y_test = x[test_idx, :], y[test_idx, :].ravel()

    # not preprocessed data
    x_train_org, y_train_org = x_org[train_idx, :], y_org[train_idx, :].ravel()
    x_val_org, y_val_org = x_org[val_idx, :], y_org[val_idx, :].ravel()
    x_test_org, y_test_org = x_org[test_idx, :], y_org[test_idx, :].ravel()

    print("Data splits shape: \n",
          "\t Train:", x_train.shape, y_train.shape, "\n",
          "\t Val:", x_val.shape, y_val.shape, "\n",
          "\t Test:", x_test.shape, y_test.shape,  "\n",
          "*******************************************************************************************", "\n",
          "x_train: \n", x_train[:5, :], "\n",
          "*******************************************************************************************", "\n",
          "y_train: \n", y_train[:5],  "\n",
          "*******************************************************************************************",
          )

    if target_is_org == 1.:
        return x_train, y_train_org, x_val, y_val_org, x_test, y_test_org
    else:
        return x_train, y_train, x_val, y_val, x_test, y_test,



