import sys
import pandas as pd
sys.path.append("../codes")
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
    if data_name == "dd_demo":

        demo_xls = pd.ExcelFile("../data/demo.xlsx")
        indicators = ['SubjID', ]

        data["normal"] = util.remove_missing_data(
            df=pd.read_excel(demo_xls, "norm").sort_values(by=indicators)
        )

        data["normal"].replace({"fem": 1, "f": 1, "masc": -1, "m": -1}, inplace=True)

        data["abnormal"] = util.remove_missing_data(
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

        data["normal"] = util.remove_missing_data(
            df=pd.read_excel(ia_report_xls, "norm").sort_values(
                by=indicators,
                axis=0)
        )

        data["abnormal"] = util.remove_missing_data(
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

        data["normal"] = util.remove_missing_data(
            df=pd.read_excel(fixation_xls, "norm").sort_values(
                by=indicators,
                axis=0)
        )

        data["abnormal"] = util.remove_missing_data(
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
