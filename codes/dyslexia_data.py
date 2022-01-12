import numpy as np
import pandas as pd
import utilities as util


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

        if "regression" in group.lower():
            targets = ['Reading_speed', ]
        else:
            targets = ['Group', ]

        all_targets = ['Reading_speed', 'Group']

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

        if "regression" in group.lower():
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

    print("Check data for NaNs or Inf: \n",
          "x: ",  np.where(x == np.inf), np.where(x == np.nan), "\n",
          "y: ", np.where(y == np.inf), np.where(y == np.nan),
          )

    return data_org, x, y, features, targets, indicators


def remove_missing_data(df):
    for col in df.columns:
        try:
            df[col].replace({".": np.nan}, inplace=True)
        except Exception as e:
            print(e, "\n No missing values in", col)

    return df.dropna()


def preprocess_data(x, y, pp):

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

    return x, y

