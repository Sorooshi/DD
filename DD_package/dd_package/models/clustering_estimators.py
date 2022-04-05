import time
import numpy as np
from skopt import BayesSearchCV
from collections import defaultdict
import dd_package.common.utils as util
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from skopt.space import Real, Categorical, Integer
from sklearn.cluster import KMeans, SpectralClustering, \
    AffinityPropagation, MeanShift, DBSCAN, AgglomerativeClustering


class ClusteringEstimators:

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

        # Methods based on given n_clusters:
        # K-Means:
        if self.estimator_name == "km_clu":
            self.tuning_estimator = KMeans(
                n_clusters=self.configs.n_clusters
            )

            # define search space
            self.params = defaultdict()
            self.params["init"] = Categorical(["k-means++", "random"])
            self.params["max_iter"] = Integer(10, 1000, "uniform")
            self.params["n_init"] = Categorical([5, 10, ])

            print (
                "Instantiate K-Means Clustering."
            )

        # Gaussian Mixture:
        elif self.estimator_name == "gm_clu":
            self.tuning_estimator = GaussianMixture(
                n_components=self.configs.n_clusters,
            )

            # define search space
            self.params = defaultdict()
            self.params["covariance_type"] = Categorical(["full", "tied", "diag", "spherical", ])
            self.params["max_iter"] = Integer(10, 1000, "uniform")
            # self.params["init_params"] = Categorical(["k-means++", "random"])

            print(
                "Instantiate Gaussian Mixture Clustering."
            )

        # Spectral (no predict() method to tune hyperparams):
        elif self.estimator_name == "s_clu":
            self.tuning_estimator = SpectralClustering(
                n_clusters=self.configs.n_clusters,
            )
            # define search space
            self.params = defaultdict()
            # self.params["n_init"] = Categorical([5, 10, ])
            # self.params["gamma"] = Real(1e-1, 1, "uniform")
            # self.params["affinity"] = Categorical(["nearest_neighbors", "rbf", ])
            # self.params["n_neighbors"] = Real(5, 20, "uniform")
            # No predict() method, thus can not be tuned using BayesSearch Opt.
            self.tuned_params = defaultdict()
            self.tuned_params["n_init"] = 10
            self.tuned_params["gamma"] = 1.0
            self.tuned_params["affinity"] = "rbf"

            print(
                "Instantiate Spectral Clustering."
            )

        # Agglomerative (no predict() method to tune hyperparams):
        elif self.estimator_name == "a_clu":
            self.tuning_estimator = AgglomerativeClustering(
                n_clusters=self.configs.n_clusters,
            )

            # define search space
            self.params = defaultdict()
            # self.params["affinity"] = Categorical(["l1", "l2", "manhattan", "cosine", ])
            # self.params["linkage"] = Categorical(["ward", "complete", "average", "single"])

            # No predict() method, thus can not be tuned using BayesSearch Opt.
            self.tuned_params = defaultdict()
            self.tuned_params["affinity"] = "l2"
            self.tuned_params["linkage"] = "average"


            print(
                "Instantiate Agglomerative Clustering."
            )

        # Methods based on automatic determination of n_clusters:
        # Affinity Propagation:
        elif self.estimator_name == "ap_clu":
            self.tuning_estimator = AffinityPropagation()

            # define search space
            self.params = defaultdict()
            self.params["max_iter"] = Integer(10, 1000, "uniform")
            self.params["damping"] = Real(5e-1, 1, "uniform")
            print(
                "Instantiate Affinity Propagation Clustering."
            )

        # DBSCAN (no predict() method to tune hyperparams):
        elif self.estimator_name == "dbs_clu":
            self.tuning_estimator = DBSCAN()

            # define search space
            self.params = defaultdict()
            # self.params["eps"] = Real(1e-1, 9e-1, "uniform")
            # self.params["min_samples"] = Integer(2, 10, "uniform")
            # self.params["p"] = Real(1, 10, "uniform")

            # No predict() method, thus can not be tuned using BayesSearch Opt.
            self.tuned_params = defaultdict()
            self.tuned_params["eps"] = 5e-1
            self.tuned_params["min_samples"] = 8
            self.tuned_params["p"] = 2

            print(
                "Instantiate DBSCAN Clustering."
            )

        # MeanShift:
        elif self.estimator_name == "ms_clu":
            self.tuning_estimator = MeanShift()

            # define search space
            self.params = defaultdict()
            self.params["max_iter"] = Integer(10, 1000, "uniform")

            print(
                "Instantiate MeanShift Clustering."
            )

        else:
            assert False, "Undefined clustering model."

        return None  # self.tuning_estimator, self.params

    def instantiate_fit_test_estimator(self, ):

        # Methods based on given n_clusters:
        if self.estimator_name == "km_clu" or \
                self.estimator_name == "a_clu":
            self.tuned_params["n_clusters"] = self.configs.n_clusters

        if self.estimator_name == "gm_clu" or self.estimator_name == "s_clu":
            self.tuned_params["n_components"] = self.configs.n_clusters
            self.tuned_params["n_components"] = self.configs.n_clusters

        # K-Means:
        if self.estimator_name == "km_clu":
            self.estimator = KMeans(**self.tuned_params)
            print (
                "K-Means Clustering."
            )

        # Gaussian Mixture:
        elif self.estimator_name == "gm_clu":
            self.estimator = GaussianMixture(**self.tuned_params)
            print(
                "Gaussian Mixture Clustering."
            )

        # Affinity Propagation:
        elif self.estimator_name == "ap_clu":
            self.estimator = AffinityPropagation(**self.tuned_params)
            print(
                "Affinity Propagation Clustering."
            )

        # Spectral:
        elif self.estimator_name == "s_clu":
            self.estimator = SpectralClustering(**self.tuned_params)
            print(
                "Spectral Clustering."
            )

        # Agglomerative:
        elif self.estimator_name == "a_clu":
            self.estimator = AgglomerativeClustering(**self.tuned_params)
            print(
                "Agglomerative Clustering."
            )

        # Methods based on automatic determination of n_clusters:
        # DBSCAN:
        elif self.estimator_name == "dbs_clu":
            self.estimator = DBSCAN(**self.tuned_params)
            print(
                "DBSCAN Clustering."
            )

        # MeanShift:
        elif self.estimator_name == "ms_clu":
            self.estimator = MeanShift(**self.tuned_params)
            print(
                "MeanShift Clustering."
            )

        # SEFNAC:
        elif self.estimator_name == "sefnac_clu":
            self.estimator = MeanShift(**self.tuned_params)
            print(
                "SEFNAC Clustering."
            )
        # KEFRiN:
        elif self.estimator_name == "kefrin_clu":
            self.estimator = MeanShift(**self.tuned_params)
            print(
                "KEFRiN Clustering."
            )

        else:
            assert False, "Undefined clustering model."

        return None

    def tune_hyper_parameters(self, ):
        """ estimator sklearn estimator, estimator dict of parameters. """

        if len(self.params.values()) != 0:  # search space has been defined for this estimator

            print(
                    "CV hyper-parameters tuning for " + self.estimator_name
            )

            # define the search
            search = BayesSearchCV(
                estimator=self.tuning_estimator,
                search_spaces=self.params,
                n_jobs=1, cv=self.cv,
                scoring="adjusted_rand_score",
                optimizer_kwargs={'base_estimator': 'RF'},
                verbose=1,
            )

            # perform the search
            search.fit(X=self.x, y=self.y, )

            # report the best result
            print("best score:", search.best_score_)
            print("best params:", search.best_params_)
            self.tuned_params = search.best_params_

        else:
            print(
                    "No CV hyper-parameters tuning for " + self.estimator_name
            )

        return None  # self.tuned_params, self.estimator

    def fit_test_tuned_estimator(self,):

        """ returns of dict of dicts, containing y_test and y_pred per each repeat. """

        print(
            "Fitting and testing of " + self.estimator_name
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
            y_pred_prob = None
            try:
                y_pred = self.estimator.predict(x_test)
            except:
                y_pred = self.estimator.fit_predict(x_test)
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
            score = adjusted_rand_score(y_test, y_pred)

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