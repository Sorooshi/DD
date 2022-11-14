import numpy as np
from collections import defaultdict
import dd_package.common.utils as util


class BaseLineModel:

    def __init__(self, y_train, learning_method, configs, test_size=100):
        self.y_train = y_train
        self.y_pred = np.array([])
        self.results = defaultdict()
        self.learning_method = learning_method
        self.test_size = test_size
        self.configs = configs

        self.y_test = self.y_train[
            np.random.randint(
                0, self.y_train.shape[0], size=self.test_size
            )
        ]

    def repeat_random_pred(self, ):

        for repeat in range(self.configs.n_repeats):
            k = str(repeat+1)
            self.results[k] = defaultdict()
            self.results[k]["y_pred"] = self._pred_randomly()
            self.results[k]["y_test"] = self.y_test

        return self.results

    def save_results(self,):

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

    def _pred_randomly(self,):

        mins = np.min(self.y_train, axis=0)
        maxs = np.max(self.y_train, axis=0)

        if self.learning_method == "regression":

            t = np.arange(mins, maxs, 1e3)
            self.y_pred = np.random.choice(t, self.test_size)
        else:
            self.y_pred = np.random.randint(low=1, high=4, size=self.test_size)

        return self.y_pred
