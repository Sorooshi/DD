import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict
from sklearn.model_selection import KFold, TimeSeriesSplit



class DyslexiaData:
    """" various forms of dataset(s)  """
    def __init__(self,
                 k_folds: int = 5,
                 path: Path= Path("../datasets"),
                 names: list = ["demo", "IA_report", "Fixation_report",],
                 ):

        self.path = path
        self.names = names
        self.k_folds = k_folds
        self.xlsx_file = pd.ExcelFile

        self.xlsx_demo = pd.ExcelFile(os.path.join(self.path, "demo.xlsx"))
        self.xlsx_ia = pd.ExcelFile(os.path.join(self.path, "IA_report.xlsx"))
        self.xlsx_fix = pd.ExcelFile(os.path.join(self.path, "Fixation_report.xlsx"))

        self.ia_datasets = defaultdict(list)
        self.fix_datasets = defaultdict(list)
        self.demo_datasets = defaultdict(list)

        self.ia = pd.DataFrame
        self.fix = pd.DataFrame
        self.demo = pd.DataFrame

        self.separated_datasets_kfold = None
        self.separated_datasets_kfold_ts = None

    def get_demo_datasets(self, ):

        print("Loading Demo data: ")

        for sheet in  self.xlsx_demo.sheet_names:
            tmp = pd.read_excel( self.xlsx_demo, sheet)
            tmp = self._remove_missing_data(df=tmp)
            tmp.replace({"fem": 1, "f": 1, "masc": 2, "m": 2}, inplace=True)
            tmp.replace({"norm": 1, "risk": 2, "dyslexia": 3}, inplace=True)
            tmp = tmp.astype({
                "Group": str,
                "SubjectID": str,
                "Sex": int,
                "Grade": int,
                "Age": int,
                "IQ": int,
                "Reading_speed": float,
                "Sound_detection": float,
                "Sound_change": float,
            })

            self.demo_datasets[sheet] = tmp.sort_values(by=["SubjectID"]).dropna()

            print(" ", sheet, tmp.shape)

        return self.demo_datasets

    def get_ia_datasets(self, ):

        print("Loading IA_report data: ")

        for sheet in self.xlsx_ia.sheet_names:
            tmp = pd.read_excel(self.xlsx_ia, sheet)
            tmp = self._remove_missing_data(df=tmp)
            tmp.replace({"norm": 1, "risk": 2, "dyslexia": 3, }, inplace=True)
            tmp = tmp.astype({
                "Group": int,
                "SubjectID": str,
                "Sentence_ID": int,
                "Word_Number": int,
                "QUESTION_ACCURACY": int,
                "FIXATION_COUNT": int,
                "SKIP": int,
                "TOTAL_READING_TIME": float,
                "FIRST_FIXATION_DURATION": float,
                "FIRST_FIXATION_X": float,
                "FIRST_FIXATION_Y": float,
                "FIRST_RUN_TOTAL_READING_TIME": float,
                "FIRST_SACCADE_AMPLITUDE": float,
                "REGRESSION_IN": int,
                "REGRESSION_OUT": int,
                "REGRESSION_OUT_FULL": int,
                "REGRESSION_PATH_DURATION": float,
            })

            self.ia_datasets[sheet] = tmp.sort_values(by=["SubjectID", "Sentence_ID", "Word_Number"]).dropna()

            print(" ", sheet, tmp.shape)

        return self.ia_datasets

    def get_fix_datasets(self, ):

        print("Loading Fixation report data:")

        for sheet in self.xlsx_fix.sheet_names:
            tmp = pd.read_excel(self.xlsx_fix, sheet)
            tmp = self._remove_missing_data(df=tmp)
            tmp.replace({"norm": 1, "risk": 2, "dyslexia": 3, }, inplace=True)
            tmp = tmp.astype({
                "Group": int,
                "SubjectID": str,
                "Sentence_ID": int,
                "Word_Number": int,
                "FIX_X": float,
                "FIX_Y": float,
                "FIX_DURATION": float,
            })

            self.fix_datasets[sheet] = tmp.sort_values(by=["SubjectID", "Sentence_ID", "Word_Number"]).dropna()

            print(" ", sheet, tmp.shape)

        return self.fix_datasets

    def concat_classes_demo(self, ):
        self.demo = pd.concat([v for k, v in self.demo_datasets.items()], axis=0)
        return self.demo

    def concat_classes_ia(self, ):
        self.ia = pd.concat([v for k, v in self.ia_datasets.items()], axis=0)
        return self.ia

    def concat_classes_fix(self, ):
        self.fix = pd.concat([v for k, v in self.fix_datasets.items()], axis=0)
        return self.fix

    @staticmethod
    def concat_dfs(df1, df2, subject_ids, features1, features2):

        """concatenates df1 tp df2, that is it casts df1 rows/columns to equal dimension of df2"""

        data = []

        for subject_id in subject_ids:
            tmp1 = df1.loc[(df1.SubjectID == subject_id)]
            tmp1 = tmp1.loc[:, features1]
            tmp2 = df2.loc[df2.SubjectID == subject_id]
            tmp2 = tmp2.loc[:, features2]

            n = tmp2.shape[0]
            tmp1 = pd.concat([tmp1] * n, ignore_index=True)

            # to check the dimension I copieed tmp2
            _tmp2 = tmp2.copy(deep=True)
            _tmp2[features1] = tmp1

            if _tmp2.shape[0] != tmp1.shape[0]:
                print(subject_id, "row")
                print("tmp2.shape:", tmp2.shape, n)
                print("tmp1.shape:", tmp1.shape)
                print("concated tmp2.shape:", tmp2.shape, n)

            if _tmp2.shape[1] != tmp1.shape[1] + tmp2.shape[1]:
                print(subject_id, "col")
                print("tmp2.shape:", tmp2.shape, n)
                print("tmp1.shape:", tmp1.shape)
                print("concated tmp2.shape:", tmp2.shape, n)

            data.append(_tmp2)

        return pd.concat(data)

    def _remove_missing_data(self, df):
        for col in df.columns:
            try:
                df[col].replace({".": np.nan}, inplace=True)
            except Exception as e:
                print(e, "\n No missing values in", col)

        return df.dropna()