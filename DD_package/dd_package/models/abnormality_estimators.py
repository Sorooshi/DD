import time
import numpy as np
import xgboost as xgb
from skopt import BayesSearchCV
from collections import defaultdict
from sklearn.svm import OneClassSVM
import dd_package.common.utils as util
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score
from skopt.space import Real, Categorical, Integer


class AbnormalityEstimators:

    def __init__(self, x, y, cv, data, estimator_name, configs, ):
        self.x = x  # np.ndarray, a pre-processed matrix of features/random variables.
        self.y = y  # np.ndarray, not pre-processed vector of target variables.
        self.cv = cv  # CV sklearn instance, stratified KFolds Cross_Validation generator with/without shuffles.
        self.data = data  # Dict of dicts, containing repeated train and test splits, (x, y np arrays).
        self.estimator_name = estimator_name.lower()  # str, name of estimator to select the method.
        self.configs = configs  # configuration dict, as namespace, to pass storing path, etc.

        self.estimator = None
        self.tuning_estimator = None
        self.params = defaultdict()
        if self.configs.run == 1:
            self.tuned_params = defaultdict()
        else:
            self.tuned_params = self.load_saved_tuned_params()
            print(
                "tuned params:\n",
                self.tuned_params,
                "\n"
            )

        self.results = defaultdict(defaultdict)

    def instantiate_tuning_estimator_and_parameters(self, ):

        # Support Vector machine method(s):
        if self.estimator_name == "ocs_ad":
            self.tuning_estimator = OneClassSVM()

            # define search space
            self.params = defaultdict()
            self.params["kernel"] = Categorical(["linear", "poly", "rbf", "sigmoid", ])
            self.params['degree'] = Integer(1, 3)
            self.params['gamma'] = Real(1e-1, 2.0, 'log-uniform')
            self.params['nu'] = Real(0, 1.0, 'uniform')

            print(
                "OneClass Support Vector Abnormality Detector."
            )

        # KNN method(s):
        elif self.estimator_name == "lof_ad":
            self.tuning_estimator = LocalOutlierFactor()

            # define search space
            self.params = defaultdict()
            self.params["n_neighbors"] = Integer(1, 100, )
            self.params["p"] = Real(1, 5, "uniform")
            self.params["leaf_size"] = Integer(5, 100)
            self.params['novelty'] = [True]

            print(
                "Local Outlier Factor (KNN-based) Abnormality Detector."
            )

        # Bayesian methods:

        # Ensemble learning method(s):
        elif self.estimator_name == "if_ad":
            self.tuning_estimator = IsolationForest(verbose=0, )

            # define search space
            self.params = defaultdict()
            self.params["n_estimators"] = Integer(10, 10000, )

            print(
                "Isolation Forest Abnormality Detector."
            )

        # Neural Networks method(s):

        else:
            assert False, "Undefined classification model."

        return None  # self.tuning_estimator, self.params

    def instantiate_train_test_estimator(self, ):

        # Support Vector machine method(s):
        if self.estimator_name == "ocs_ad":
            self.estimator = OneClassSVM(**self.tuned_params)
            print(
                "Instantiate OneClass Support Vector Abnormality Detector."
            )

        # KNN method(s):
        elif self.estimator_name == "lof_ad":
            self.estimator = LocalOutlierFactor(**self.tuned_params)
            print(
                "Instantiate Local Outlier Factor (KNN) Abnormality Detector."
            )

        # Bayesian methods:

        # Ensemble learning method(s):
        elif self.estimator_name == "If_ad":
            self.estimator = IsolationForest(**self.tuned_params)

            print(
                "Instantiate Isolation Forest Abnormality Detector."
            )

        # Neural Networks method(s):

        else:
            assert False, "Undefined classification model."

        return None  # self.estimator

    def tune_hyper_parameters(self, ):
        """ estimator sklearn estimator, estimator dict of parameters. """

        print("CV hyper-parameters tuning for " + self.estimator_name)

        # define the search
        search = BayesSearchCV(
            estimator=self.tuning_estimator,
            search_spaces=self.params,
            n_jobs=1, cv=self.cv,
            scoring="accuracy",
            optimizer_kwargs={'base_estimator': 'RF'},
            verbose=1,
        )

        # perform the search
        search.fit(X=self.x, y=self.y, )

        # report the best result
        print("best score:", search.best_score_)
        print("best params:", search.best_params_)
        self.tuned_params = search.best_params_

        return None  # self.tuned_params, self.estimator

    def train_test_tuned_estimator(self,):

        """ returns of dict of dicts, containing y_test and y_pred per each repeat. """

        print(
            "Training and testing of " + self.estimator_name
        )

        old_score = - np.inf

        for k, v in self.data.items():
            self.results[k] = defaultdict()

            run = util.init_a_wandb(
                name=self.configs.name_wb,
                project=self.configs.project,
                notes="--",
                group=self.configs.group,
                tag=[self.configs.data_name],
                config=self.tuned_params,
            )

            start = time.time()

            self.estimator.fit(v["x_train"], v["y_train"])
            y_test = np.asarray([1 if i == 1 else -1 for i in v["y_test"]])
            x_test = v["x_test"]
            y_pred = self.estimator.predict(x_test)

            try:
                y_pred_prob = self.estimator.predict_proba(x_test)
            except:
                y_pred_prob = self.estimator.decision_function(x_test)

            end = time.time()

            self.results[k]["y_test"] = y_test
            self.results[k]["x_test"] = x_test
            self.results[k]["y_pred"] = y_pred
            self.results[k]["y_pred_prob"] = y_pred_prob
            self.results[k]["exe_time"] = end - start

            run = util.wandb_metrics(
                run=run,
                y_true=y_test,
                y_pred=y_pred,
                y_pred_prob=y_pred_prob,
                learning_method=self.configs.learning_method,
            )

            # to save the best results model and plots
            score = accuracy_score(y_test, y_pred)

            if score > old_score:
                old_score = score

                run = util.wandb_true_pred_plots(
                    run=run, y_true=y_test, y_pred=y_pred,
                    path=self.configs.figures_path,
                    specifier=self.configs.specifier,
                )

                run = util.wandb_true_pred_scatters(
                    run=run, y_test=y_test, y_pred=y_pred,
                    path=self.configs.figures_path,
                    specifier=self.configs.specifier,
                )

                run = util.wandb_true_pred_histograms(
                    run=run, y_test=y_test, y_pred=y_pred,
                    path=self.configs.figures_path,
                    specifier=self.configs.specifier,
                )

                _ = util.save_model(
                    path=self.configs.models_path,
                    model=self.estimator,
                    specifier=self.configs.specifier,
                )

            run.finish()

        return None  # self.results

    def save_params_results(self,):
        # save tuned_params
        util.save_a_dict(
            a_dict=self.tuned_params,
            name=self.configs.specifier,
            save_path=self.configs.params_path,
        )

        # save results
        util.save_a_dict(
            a_dict=self.results,
            name=self.configs.specifier,
            save_path=self.configs.results_path,
        )

        return None

    def load_saved_tuned_params(self,):
        saved_tuned_params = util.load_a_dict(
            name=self.configs.specifier,
            save_path=self.configs.params_path
        )
        return saved_tuned_params

    def print_results(self, ):

        results = util.load_a_dict(
            name=self.configs.specifier,
            save_path=self.configs.results_path,
        )

        util.print_the_evaluated_results(
            results,
            self.configs.learning_method,
        )

        return None




















