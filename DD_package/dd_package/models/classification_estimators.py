import time
import numpy as np
import xgboost as xgb
from sklearn.svm import SVC
from skopt import BayesSearchCV
from collections import defaultdict
import dd_package.common.utils as util
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from skopt.space import Real, Categorical, Integer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, \
    ExpSineSquared, ConstantKernel, RBF
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB


class ClassificationEstimators:

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
        if self.estimator_name == "l_cls":
            self.tuning_estimator = LogisticRegression(
                solver="saga",
                multi_class="multinomial",
            )

            # define search space
            self.params = defaultdict()
            self.params["penalty"] = Categorical(["none", "l1", "l2", "elasticnet", ])
            self.params['C'] = Real(1e-1, 4.0, 'log-uniform')
            self.params["fit_intercept"] = Categorical([True, False])
            self.params["max_iter"] = Integer(100, 100000, "uniform")
            self.params["l1_ratio"] = Real(1e-1, 9e-1, "uniform")
            # self.params["solver"] = "saga"  # These solvers: "newton-cg", "sag", "lbfgs", don't support all penalties.
            # self.params["multi_class"] = "multinomial"  # to use cross-entropy loss in all cases

            print (
                "Logistic Classifier."
            )

        # Support Vector machine method(s):
        elif self.estimator_name == "sv_cls":
            self.tuning_estimator = SVC()

            # define search space
            self.params = defaultdict()
            self.params["kernel"] = Categorical(["linear", "poly", "rbf", "sigmoid", ])
            self.params['degree'] = Integer(1, 3)
            self.params['C'] = Real(1e-1, 4.0, 'log-uniform')
            self.params['gamma'] = Real(1e-1, 2.0, 'log-uniform')

            print(
                "C Support Vector Classifier."
            )

        # KNN method(s):
        elif self.estimator_name == "knn_cls":
            self.tuning_estimator = KNeighborsClassifier()

            # define search space
            self.params = defaultdict()
            self.params["n_neighbors"] = Integer(1, 10, )
            self.params["p"] = Real(1, 5, "uniform")

            print(
                "KNearest Neighbor Classifier."
            )

        # Bayesian methods:
        elif self.estimator_name == "gnb_cls":
            self.tuning_estimator = GaussianNB()

            # define search space
            self.params = defaultdict()
            self.params["var_smoothing"] = Real(1e-9, 1e-7, "uniform")

            print(
                "Instantiate Naive Gaussian Bayese Classifier."
            )

        elif self.estimator_name == "mnb_cls":
            self.tuning_estimator = MultinomialNB()

            # define search space
            self.params = defaultdict()
            self.params["alpha"] = Real(1e-3, 1, "uniform")

            print(
                "Instantiate Naive Bayes: Multinomial Classifier."
            )

        elif self.estimator_name == "cnb_cls":
            self.tuning_estimator = ComplementNB()

            # define search space
            self.params = defaultdict()
            self.params["alpha"] = Real(1e-3, 1, "uniform")

            print(
                "Instantiate Naive Bayes: Complement Classifier."
            )

        # Ensemble learning method(s):
        elif self.estimator_name == "rf_cls":
            self.tuning_estimator = RandomForestClassifier(verbose=0, )

            # define search space
            self.params = defaultdict()
            self.params["n_estimators"] = Integer(10, 10000, )
            self.params["min_samples_split"] = Integer(2, 10, )
            self.params["min_samples_leaf"] = Integer(1, 10, )

            print(
                "Random Forest Classifier."
            )

        elif self.estimator_name == "gb_cls":
            self.tuning_estimator = GradientBoostingClassifier(verbose=0, )

            # define search space
            self.params = defaultdict()
            self.params["learning_rate"] = Real(1e-3, 5e-1, "uniform")
            self.params["n_estimators"] = Integer(10, 10000, )
            self.params["min_samples_split"] = Integer(2, 10, )
            self.params["min_samples_leaf"] = Integer(1, 10, )

            print(
                "Gradient Boosting Classifier."
            )

        elif self.estimator_name == "ab_cls":
            self.tuning_estimator = AdaBoostClassifier()

            # define search space
            self.params = defaultdict()
            self.params["n_estimators"] = Integer(10, 10000, )
            self.params["learning_rate"] = Real(1e-3, 5e-1, "uniform")
            print(
                "Adaboost Classifier."
            )

        elif self.estimator_name == "xgb_cls":
            self.tuning_estimator = xgb.XGBClassifier()

            # define search space
            self.params = defaultdict()
            self.params["n_estimators"] = Integer(10, 10000, )
            self.params["learning_rate"] = Real(1e-3, 5e-1, "uniform")
            self.params["max_depth"] = Integer(1, 100, "uniform")

            print(
                "XGBoost Classifier."
            )

        # Gaussian Process method(s):
        elif self.estimator_name == "_gp_cls_+":
            self.tuning_estimator = GaussianProcessClassifier()
            # Previously we faced some issue due to limits of
            #   GP due the dataset size, and thus for now I won't consider it.

            kernel_rbf = RBF(
                length_scale=1.0, length_scale_bounds=(1e-1, 10.0)
            )

            kernel_rational_quadratic = RationalQuadratic(
                length_scale=1.0, alpha=0.1, alpha_bounds=(1e-5, 1e15)
            )

            kernel_exp_sin_squared = ExpSineSquared(
                length_scale=1.0,
                periodicity=3.0,
                length_scale_bounds=(0.1, 10.0),
                periodicity_bounds=(1.0, 10.0),
            )

            kernel = ConstantKernel(
                constant_value=1.0, constant_value_bounds=(0.0, 10.0)
            ) * RBF(
                length_scale=0.5, length_scale_bounds=(0.0, 10.0)) + RBF(
                length_scale=2.0, length_scale_bounds=(0.0, 10.0))

            # define search space
            self.params = defaultdict()
            self.params["kernel"] = Categorical([kernel, None])  # how to define?
            # Categorical([kernel_rbf, kernel_rational_quadratic, kernel_exp_sin_squared])
            print(
                "Gaussian Process Classifier."
            )

        # Neural Networks method(s):
        elif self.estimator_name == "mlp_cls":
            self.tuning_estimator = MLPClassifier(
                shuffle=False,
                verbose=False,
            )

            # define search space
            self.params = defaultdict()
            self.params["hidden_layer_sizes"] = (2, 200,)
            self.params["activation"] = Categorical(["identity", "logistic", "tanh", "relu"])
            self.params["solver"] = Categorical(["lbfgs", "sgd", "adam"])
            # self.params["alpha"] = Real(1e-6, 1e-2, "uniform")
            # self.params["learning_rate"] = Categorical(["constant", "invscaling", "adaptive"])
            # self.params["learning_rate_init"] = Real(1e-4, 1e-1, "uniform")
            self.params["max_iter"] = Integer(100, 50000, "uniform")

            print(
                "Multi Layer Perceptron Classifier."
            )

        else:
            assert False, "Undefined classification model."

        return None  # self.tuning_estimator, self.params

    def instantiate_train_test_estimator(self, ):

        # Simplest learning method(s):
        if self.estimator_name == "l_cls":
            self.tuned_params["solver"] = "saga"
            self.tuned_params["multi_class"] = "multinomial"
            self.estimator = LogisticRegression(**self.tuned_params)
            print (
                "Instantiate Logistic Classifier. \n",
                self.tuned_params
            )

        # Support Vector machine method(s):
        elif self.estimator_name == "sv_cls":
            self.estimator = SVC(**self.tuned_params)
            print(
                "Instantiate C Support Vector Classifier."
            )

        # KNN method(s):
        elif self.estimator_name == "knn_cls":
            self.estimator = KNeighborsClassifier(**self.tuned_params)
            print(
                "Instantiate KNearest Neighbor Classifier."
            )

        # Bayesian methods:
        elif self.estimator_name == "gnb_cls":
            self.estimator = GaussianNB(**self.tuned_params)

            print(
                "Instantiate Naive Gaussian Bayese Classifier."
            )

        elif self.estimator_name == "mnb_cls":
            self.estimator = MultinomialNB(**self.tuned_params)
            print(
                "Instantiate Naive Bayes: Multinomial Classifier."
            )

        elif self.estimator_name == "cnb_cls":
            self.estimator = ComplementNB(**self.tuned_params)

            print(
                "Instantiate Naive Bayes: Complement Classifier."
            )

        # Ensemble learning method(s):
        elif self.estimator_name == "rf_cls":
            self.estimator = RandomForestClassifier(**self.tuned_params)

            print(
                "Instantiate Random Forest Classifier."
            )

        elif self.estimator_name == "gb_cls":
            self.estimator = GradientBoostingClassifier(**self.tuned_params)
            print(
                "Instantiate Gradient Boosting Classifier."
            )

        elif self.estimator_name == "ab_cls":  # does not support 2d y
            self.estimator = AdaBoostClassifier(**self.tuned_params)
            print(
                "Instantiate Adaboost Classifier."
            )

        elif self.estimator_name == "xgb_cls":
            self.estimator = xgb.XGBClassifier(**self.tuned_params)
            print(
                "Instantiate XGBoost Classifier."
            )

        # Gaussian Process method(s):
        elif self.estimator_name == "gp_cls":
            self.estimator = GaussianProcessClassifier(**self.tuned_params)
            # Previously we faced some issue due to limits of
            #   GP due dataset size, and thus for now I won't consider it
            print(
                "Instantiate Gaussian Process Classifier."
            )

        # Neural Networks method(s):
        elif self.estimator_name == "mlp_cls":
            self.estimator = MLPClassifier(**self.tuned_params)
            print(
                "Instantiate Multi-Layer Perceptron Classifier."
            )

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
            y_test = v["y_test"]
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

























