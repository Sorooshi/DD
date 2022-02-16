from sklearn.svm import SVR
from skopt import BayesSearchCV
from collections import defaultdict
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from skopt.space import Real, Categorical, Integer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor


class RegressionEstimators:
    def __init__(self, x, y, cv, data, estimator_name, ):
        self.x = x  # np.ndarray, a pre-processed matrix of features/random variables.
        self.y = y  # np.ndarray, not pre-processed vector of target variables.
        self.cv = cv  # CV sklearn instance, stratified KFolds Cross_Validation generator with/without shuffles.
        self.data = data  # Dict of dicts, containing repeated train and test splits, (x, y np arrays).
        self.estimator_name = estimator_name.lower()  # str, name of estimator to select the method.


        self.estimator = None
        self.params = defaultdict()

        self.results = defaultdict(defaultdict)

    def instantiate_an_estimator_and_parameters(self,):

        # Simplest learning method(s):
        if self.estimator_name == "l_reg":
            self.estimator = LinearRegression()

            # define search space
            self.params = defaultdict()
            self.params["fit_intercept"] = Categorical([True, False])

            print ("Linear Regressor.")

        # Support Vector machine method(s):
        elif self.estimator_name == "sv_reg":
            self.estimator = SVR()

            # define search space
            self.params = defaultdict()
            self.params["kernel"] = Categorical(["linear", "poly", "rbf", "sigmoid", ])
            self.params['degree'] = Integer(1, 3)
            self.params['C'] = Real(1e-1, 4.0, 'log-uniform')
            self.params['gamma'] = Real(1e-1, 2.0, 'log-uniform')
            self.params["epsilon"] = Real(1e-1, 2.0, 'log-uniform')

            print("Linear Support Vector Regression.")

        # KNN method(s):
        elif self.estimator_name == "knn_reg":
            self.estimator = KNeighborsRegressor()

            # define search space
            self.params = defaultdict()
            self.params["n_neighbors"] = Integer(1, 10,)
            self.params["p"] = Real(1, 5, "uniform")

            print("KNearest Neighbor Regressor.")

        # Ensemble learning method(s):
        elif self.estimator_name == "rf_reg":
            self.estimator = RandomForestRegressor(verbose=1,)

            # define search space
            self.params = defaultdict()
            self.params["n_estimators"] = Integer(10, 1000, )
            self.params["min_samples_split"] = Integer(2, 10, )
            self.params["min_samples_leaf"] = Integer(1, 10, )

            print("Random Forest Regressor.")

        elif self.estimator_name == "gb_reg":
            self.estimator = GradientBoostingRegressor(verbose=1, )

            # define search space
            self.params = defaultdict()
            self.params["loss"] = Categorical(["squared_error", "absolute_error", "huber", "quantile"])
            self.params["learning_rate"] = Real(1e-3, 5e-1, "uniform")
            self.params["n_estimators"] = Integer(10, 1000, )
            self.params["min_samples_split"] = Integer(2, 10, )
            self.params["min_samples_leaf"] = Integer(1, 10, )
            self.params["alpha"] = Real(1e-1, 9e-1, "Uniform")

            print("Gradient Boosting Regressor.")

        elif self.estimator_name == "a_reg":  # does not support 2d y
            self.estimator = AdaBoostRegressor()

            # define search space
            self.params = defaultdict()
            self.params["n_estimators"] = Integer(10, 1000,)
            self.params["learning_rate"] = Real(1e-3, 5e-1, "uniform")

            print("Adaboost Regressor.")

        # Gaussian Process method(s):
        elif self.estimator_name == "gp_reg":
            self.estimator = GaussianProcessRegressor()
            # Previously we faced some issue due to limits of
            #   GP due dataset size, and thus for now I won't consider it
            print("Gaussian Process Regressor.")

        # Neural Networks method(s):
        elif self.estimator_name == "mlp_reg":
            self.estimator = MLPRegressor(
                shuffle=False,
                verbose=True,
            )

            # define search space
            self.params = defaultdict()
            self.params["hidden_layer_sizes"] = (2, 200, )
            self.params["activation"] = Categorical(["identity", "logistic", "tanh", "relu"])
            self.params["solver"] = Categorical(["lbfgs", "sgd", "adam"])
            self.params["alpha"] = Real(1e-6, 1e-2, "uniform")
            self.params["learning_rate"] = Categorical(["constant", "invscaling", "adaptive"])
            self.params["learning_rate_init"] = Real(1e-4, 1e-2, "uniform")
            self.params["max_iter"] = Real(100, 2000, "uniform")

            print("Multi Layer Perceptron Regressor.")

        else:
            print ("Undefined regression model.")
            f = True
            assert f is True

        return self.estimator, self.params

    def tune_hyper_parameters(self, estimator, params, ):
        """ estimator sklearn estimator, estimator dict of parameters. """

        print("CV yyperparameters tuning of" + self.estimator_name)

        # define the search
        search = BayesSearchCV(estimator=estimator,
                               search_spaces=params,
                               n_jobs=-2, cv=self.cv,
                               )
        # perform the search
        search.fit(X=self.x, y=self.y, )

        # report the best result
        print("best score:", search.best_score_)
        print("best params:", search.best_params_)
        tuned_params = search.best_params_

        return tuned_params

    def train_test_tuned_estimator(self, estimator, tuned_params):

        """ returns of dict of dicts, containing y_test and y_pred per each repeat. """

        print("Training and testing" + self.estimator_name)
        estimator = estimator(**tuned_params)

        for k, v in self.data.items():
            self.results[k] = defaultdict()
            for kk, vv in v.items():
                estimator.fit(vv["x_train"], vv["y_train"])
                self.results[k]["y_pred"] = estimator.predict(vv["x_test"])
                self.results[k]["y_true"] = vv["y_test"]

        return self.results























