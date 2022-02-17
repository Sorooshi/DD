import numpy as np
from sklearn.svm import SVR
from skopt import BayesSearchCV
from collections import defaultdict
from sklearn.metrics import r2_score
import dd_package.common.utils as util
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from skopt.space import Real, Categorical, Integer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge


class RegressionEstimators:

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
        self.tuned_params = defaultdict()

        self.results = defaultdict(defaultdict)

    def instantiate_tuning_estimator_and_parameters(self, ):

        # Simplest learning method(s):
        if self.estimator_name == "l_reg":
            self.tuning_estimator = LinearRegression()

            # define search space
            self.params = defaultdict()
            self.params["fit_intercept"] = Categorical([True, False])

            print (
                "Linear Regressor."
            )

        # Support Vector machine method(s):
        elif self.estimator_name == "sv_reg":
            self.tuning_estimator = SVR()

            # define search space
            self.params = defaultdict()
            self.params["kernel"] = Categorical(["linear", "poly", "rbf", "sigmoid", ])
            self.params['degree'] = Integer(1, 3)
            self.params['C'] = Real(1e-1, 4.0, 'log-uniform')
            self.params['gamma'] = Real(1e-1, 2.0, 'log-uniform')
            self.params["epsilon"] = Real(1e-1, 2.0, 'log-uniform')

            print(
                "Linear Support Vector Regression."
            )

        # KNN method(s):
        elif self.estimator_name == "knn_reg":
            self.tuning_estimator = KNeighborsRegressor()

            # define search space
            self.params = defaultdict()
            self.params["n_neighbors"] = Integer(1, 10, )
            self.params["p"] = Real(1, 5, "uniform")

            print(
                "KNearest Neighbor Regressor."
            )

        # Bayesian Ridge:
        elif self.estimator_name == "br_reg":
            self.tuning_estimator = BayesianRidge()

            # define search space
            self.params = defaultdict()
            self.params["n_iter"] = Integer(1e2, 2e4, "uniform")
            self.params["alpha_1"] = Real(1e-8, 1e-2, "uniform")
            self.params["alpha_2"] = Real(1e-8, 1e-2, "uniform")
            self.params["lambda_1"] = Real(1e-8, 1e-2, "uniform")
            self.params["lambda_2"] = Real(1e-8, 1e-2, "uniform")
            self.params["fit_intercept"] = Categorical([True, False])

            print(
                "Instantiate Bayesian Ridge Regressor."
            )

        # Ensemble learning method(s):
        elif self.estimator_name == "rf_reg":
            self.tuning_estimator = RandomForestRegressor(verbose=1, )

            # define search space
            self.params = defaultdict()
            self.params["n_estimators"] = Integer(10, 1000, )
            self.params["min_samples_split"] = Integer(2, 10, )
            self.params["min_samples_leaf"] = Integer(1, 10, )

            print(
                "Random Forest Regressor."
            )

        elif self.estimator_name == "gb_reg":
            self.tuning_estimator = GradientBoostingRegressor(verbose=1, )

            # define search space
            self.params = defaultdict()
            # self.params["loss"] = Categorical(["squared_error", "absolute_error", "huber", "quantile"])
            self.params["learning_rate"] = Real(1e-3, 5e-1, "uniform")
            self.params["n_estimators"] = Integer(10, 1000, )
            self.params["min_samples_split"] = Integer(2, 10, )
            self.params["min_samples_leaf"] = Integer(1, 10, )
            self.params["alpha"] = Real(1e-1, 9e-1, "uniform")

            print(
                "Gradient Boosting Regressor."
            )

        elif self.estimator_name == "ab_reg":
            self.tuning_estimator = AdaBoostRegressor()

            # define search space
            self.params = defaultdict()
            self.params["n_estimators"] = Integer(10, 1000, )
            self.params["learning_rate"] = Real(1e-3, 5e-1, "uniform")

            print(
                "Adaboost Regressor."
            )

        # Gaussian Process method(s):
        elif self.estimator_name == "gp_reg":
            self.tuning_estimator = GaussianProcessRegressor()
            # Previously we faced some issue due to limits of
            #   GP due dataset size, and thus for now I won't consider it
            print(
                "Gaussian Process Regressor."
            )

        # Neural Networks method(s):
        elif self.estimator_name == "mlp_reg":
            self.tuning_estimator = MLPRegressor(
                shuffle=False,
                verbose=True,
            )

            # define search space
            self.params = defaultdict()
            self.params["hidden_layer_sizes"] = (2, 200,)
            self.params["activation"] = Categorical(["identity", "logistic", "tanh", "relu"])
            self.params["solver"] = Categorical(["lbfgs", "sgd", "adam"])
            self.params["alpha"] = Real(1e-6, 1e-2, "uniform")
            self.params["learning_rate"] = Categorical(["constant", "invscaling", "adaptive"])
            self.params["learning_rate_init"] = Real(1e-4, 1e-1, "uniform")
            self.params["max_iter"] = Real(100, 10000, "uniform")

            print(
                "Multi Layer Perceptron Regressor."
            )

        else:
            assert False, "Undefined regression model."

        return None  # self.tuning_estimator, self.params

    def instantiate_train_test_estimator(self, ):

        # Simplest learning method(s):
        if self.estimator_name == "l_reg":
            self.estimator = LinearRegression(**self.tuned_params)
            print (
                "Instantiate Linear Regressor."
            )

        # Support Vector machine method(s):
        elif self.estimator_name == "sv_reg":
            self.estimator = SVR(**self.tuned_params)
            print(
                "Instantiate Linear Support Vector Regression."
            )

        # KNN method(s):
        elif self.estimator_name == "knn_reg":
            self.estimator = KNeighborsRegressor(**self.tuned_params)
            print(
                "Instantiate KNearest Neighbor Regressor."
            )

        # Bayesian Ridge:
        elif self.estimator_name == "br_reg":
            self.estimator = BayesianRidge(**self.tuned_params)
            print(
                "Instantiate Bayesian Ridge Regressor."
            )

        # Ensemble learning method(s):
        elif self.estimator_name == "rf_reg":
            self.estimator = RandomForestRegressor(**self.tuned_params)

            print(
                "Instantiate Random Forest Regressor."
            )

        elif self.estimator_name == "gb_reg":
            self.estimator = GradientBoostingRegressor(**self.tuned_params)
            print(
                "Instantiate Gradient Boosting Regressor."
            )

        elif self.estimator_name == "ab_reg":  # does not support 2d y
            self.estimator = AdaBoostRegressor(**self.tuned_params)
            print(
                "Instantiate Adaboost Regressor."
            )

        # Gaussian Process method(s):
        elif self.estimator_name == "gp_reg":
            self.estimator = GaussianProcessRegressor(**self.tuned_params)
            # Previously we faced some issue due to limits of
            #   GP due dataset size, and thus for now I won't consider it
            print(
                "Instantiate Gaussian Process Regressor."
            )

        # Neural Networks method(s):
        elif self.estimator_name == "mlp_reg":
            self.estimator = MLPRegressor(**self.tuned_params)
            print(
                "Instantiate Multi-Layer Perceptron Regressor."
            )

        else:
            assert False, "Undefined regression model."

        return None  # self.estimator

    def tune_hyper_parameters(self, ):
        """ estimator sklearn estimator, estimator dict of parameters. """

        print("CV hyper-parameters tuning for " + self.estimator_name)

        # define the search
        search = BayesSearchCV(estimator=self.tuning_estimator,
                               search_spaces=self.params,
                               n_jobs=-2, cv=self.cv,
                               scoring="r2",
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

            self.estimator.fit(v["x_train"], v["y_train"])
            y_test = v["y_test"]
            y_pred = self.estimator.predict(v["x_test"])
            self.results[k]["y_pred"] = y_pred
            self.results[k]["y_test"] = y_test

            run = util.wandb_metrics(
                run=run,
                y_true=y_test,
                y_pred=y_pred,
                learning_method=self.configs.learning_method,
            )

            # to save the best results model and plots
            score = r2_score(y_test, y_pred)

            if score > old_score:
                old_score = score

                run = util.wandb_true_pred_plots(
                    run=run, y_true=y_test, y_pred=y_pred,
                    path=self.configs.figures_path,
                    specifier=self.configs.specifier+"-"+k,
                )

                run = util.wandb_true_pred_scatters(
                    run=run, y_test=y_test, y_pred=y_pred,
                    path=self.configs.figures_path,
                    specifier=self.configs.specifier+"-"+k,
                )

                run = util.wandb_true_pred_histograms(
                    run=run, y_test=y_test, y_pred=y_pred,
                    path=self.configs.figures_path,
                    specifier=self.configs.specifier+"-"+k,
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

    def print_results(self, ):

        # no tuning or training has been executed
        if len(self.results.values()) != 0:
            util.print_the_evaluated_results(
                self.results,
                self.configs.learning_method,
            )

        else:
            results = util.load_a_dict(
                name=self.configs.specifier,
                save_path=self.configs.results_path,
            )

            util.print_the_evaluated_results(
                results,
                self.configs.learning_method,
            )

        return None

























