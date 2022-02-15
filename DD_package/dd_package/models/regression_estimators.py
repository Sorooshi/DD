from skopt import BayesSearchCV
from collections import defaultdict
from sklearn.svm import LinearSVR
from types import SimpleNamespace
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor


class RegressionEstimators:
    def __init__(self, run, group, cv, data, x, y, estimator_name, learning_method,):
        self.run = run
        self.group = group  # FCluster, TCluster, ... ?
        self.estimator_name = estimator_name.lower()
        self.learning_method = learning_method.lower()
        self.cv = cv  # stratified KFolds Cross_Validation generator with/without shuffles
        self.data = data  # Dict of dicts, containing repeated train and test splits, (x, y np arrays).
        self.x = x  # features/random variables
        self.y = y  # target variables

        self.estimator = None

        self.results = defaultdict(defaultdict)

    def instantiate_an_estimator(self,):

        # Simplest learning method(s):
        if self.estimator_name == "lr":
            self.estimator = LinearRegression()
            print ("Linear Regressor.")

        # Support Vector machine method(s):
        elif self.estimator_name == "lsvr":
            self.estimator = LinearSVR()
            print("Linear Support Vector Regression.")

        # KNN method(s):
        elif self.estimator_name == "knr":
            self.estimator = KNeighborsRegressor()
            print("KNearest Neighbor Regressor.")

        # Ensemble learning method(s):
        elif self.estimator_name == "rfr":
            self.estimator = RandomForestRegressor(verbose=1,)
            print("Random Forest Regressor.")
        elif self.estimator_name == "gbr":
            self.estimator = GradientBoostingRegressor(verbose=1, )
            print("Gradient Boosting Regressor.")
        elif self.estimator_name == "ar":  # does not support 2d y
            self.estimator = AdaBoostRegressor()
            print("Adaboost Regressor.")

        # Gaussian Process method(s):
        elif self.estimator_name == "gpr":
            self.estimator = GaussianProcessRegressor()
            print("Gaussian Process Regressor.")

        # Neural Networks method(s):
        elif self.estimator_name == "mlpr":
            self.estimator = MLPRegressor()
            print("Multi Layer Perceptron Regressor.")
        else:
            print ("Undefined regression model.")
            f = True
            assert f is True

        return self.estimator

    def tune_hyper_parameters(self, estimator, params, ):
        """ estimator sklearn estimator, estimator dict of parameters. """

        print("CV yyperparameters tuning of" + self.estimator_name)

        # define the search
        search = BayesSearchCV(estimator=estimator,
                               search_spaces=params,
                               n_jobs=-2, cv=self.cv,
                               )
        # perform the search
        search.fit(X=self.x.values, y=self.y["Reading_speed"].values, )

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























