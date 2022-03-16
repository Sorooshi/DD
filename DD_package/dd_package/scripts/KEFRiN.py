import os
import time
import pickle
import argparse
import warnings
import numpy as np
from copy import deepcopy
from sklearn import metrics
import processing_tools as pt
from sklearn.preprocessing import OneHotEncoder


warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True, linewidth=120, precision=3)

current_seed = np.random.get_state()[1][0]
with open('granary_k.txt', 'a+') as o:
    o.write(str(current_seed)+'\n')


def args_parser(args):
    name = args.Name
    run = args.Run
    rho = args.Rho
    xi = args.Xi
    n_clusters = args.N_clusters
    with_noise = args.With_noise
    pp = args.PreProcessing
    setting = args.Setting
    max_iterations = args.Max_iterations
    kmean_pp = args.Kmeans_pp
    euclidean = args.Euclidean
    cosine = args.Cosine
    manhattan = args.Manhattan
#    minkowski = args.Minkowski

    return name, run, rho, xi, n_clusters, with_noise, pp,\
           setting, max_iterations, kmean_pp, euclidean, cosine, manhattan


class EkifenMin:

    def __init__(self, y, p, rho, xi, n_clusters, kmean_pp, euclidean, cosine, manhattan, max_iteration):
        super(EkifenMin, self).__init__()
        self.y = y
        self.p = p
        self.rho = rho
        self.xi = xi
        self.n_clusters = n_clusters
        self.kmeans_pp = kmean_pp
        self.euclidean = euclidean
        self.cosine = cosine
        self.manhattan = manhattan
        self.max_iterations = max_iteration
        self.n = y.shape[0]  # number of entities/nodes/observations
        self.v = y.shape[1]  # number of features/attributes
        self.centroids_y = []  # centroid in features space (c_v)
        self.centroids_p = []  # centroid in links space (b_j)
        self.clusters_labels = np.array([]).reshape(self.n, 0)
        self.seeds = np.random.choice(np.arange(0, self.n), size=self.n_clusters, replace=False)
        self.output_y = {}
        self.output_p = {}

    @staticmethod
    def compute_euclidean(data_points, centroid):
        return np.nansum(np.power(data_points - centroid, 2), axis=1)

    @staticmethod
    def compute_euclidean_(data_point, centroid):
        return np.nansum(np.power(data_point - centroid, 2))

    @staticmethod
    def compute_cosine(data_points, centroid):
        # add a very small number (1e-10) to avoid overflow and producing nan's
        return 1-np.divide(np.inner(data_points, centroid)+1e-10,
                           (np.sqrt(np.nansum(data_points**2, axis=1)+1e-10) *
                            np.sqrt(np.nansum(centroid**2)+1e-10)))

    @staticmethod
    def compute_cosine_(data_point, centroid):
        return 1-np.divide(np.inner(data_point, centroid)+1e-10,
                           (np.sqrt(np.nansum(data_point**2)+1e-10) *
                            np.sqrt(np.nansum(centroid**2)+1e-10)))

    @staticmethod
    def compute_manhattan(data_points, centroid):
        return np.nansum(np.abs(data_points - centroid), axis=1)

    @staticmethod
    def compute_manhattan_(data_point, centroid):
        return np.nansum(np.abs(data_point - centroid))

    @staticmethod
    def compute_minkowski(data_points, centroid, p_value):
        return np.power(np.nansum(np.power(np.abs(data_points-centroid),
                                           p_value), axis=1), 1./p_value)

    def kmeans_plus_plus(self,):
        centroids_indices = [np.random.randint(0, self.y.shape[0])]
        for k in range(self.n_clusters-1):
            d_y = np.zeros([self.n])  # distances matrix
            d_p = np.zeros([self.n])  # distances matrix
            for i in range(self.y.shape[0]):
                if i not in centroids_indices:
                    if self.euclidean == 1:
                        d_y[i] = self.compute_euclidean_(self.y[i, :], self.y[centroids_indices[k], :])
                        d_p[i] = self.compute_euclidean_(self.p[i, :], self.p[centroids_indices[k], :])
                    elif self.cosine == 1:
                        d_y[i] = self.compute_cosine_(self.y[i, :], self.y[centroids_indices[k], :])
                        d_p[i] = self.compute_cosine_(self.p[i, :], self.p[centroids_indices[k], :])
                    elif self.manhattan == 1:
                        d_y[i] = self.compute_manhattan_(self.y[i, :], self.y[centroids_indices[k], :])
                        d_p[i] = self.compute_manhattan_(self.p[i, :], self.p[centroids_indices[k], :])
            d_t = self.xi*d_p + self.rho*d_y
            next_center = np.argmax(d_t)
            centroids_indices.append(next_center)

        centroid_temp_y = self.y[centroids_indices, :]
        centroid_temp_p = self.p[centroids_indices, :]

        return centroid_temp_y, centroid_temp_p

    def apply_ekifen(self):
        # Initialization
        if self.kmeans_pp == 1:
            self.centroids_y, self.centroids_p = self.kmeans_plus_plus()
        else:
            for seed in self.seeds:
                self.centroids_y.append(self.y[seed, :])
                self.centroids_p.append(self.p[seed, :])
            self.centroids_y = np.asarray(self.centroids_y)
            self.centroids_p = np.asarray(self.centroids_p)

        f_iter = True  # iteration continue flag
        c_iter = 0  # iteration counter

        while f_iter is True:
            # print("f_iter:", f_iter, "c_iter:", c_iter)
            # Computing the distances between entities and centroids.
            distances = []
            for k in range(self.centroids_y.shape[0]):
                # computing the Euclidean distance of k-th cluster center
                # with all n entries of the corresponding matrices
                if self.euclidean == 1:
                    tmp_distance_y = self.compute_euclidean(self.y, self.centroids_y[k, :])
                    tmp_distance_p = self.compute_euclidean(self.p, self.centroids_p[k, :])

                elif self.cosine == 1:
                    tmp_distance_y = self.compute_cosine(self.y, self.centroids_y[k, :])
                    tmp_distance_p = self.compute_cosine(self.p, self.centroids_p[k, :])

                elif self.manhattan == 1:
                    tmp_distance_y = self.compute_manhattan(self.y, self.centroids_y[k, :])
                    tmp_distance_p = self.compute_manhattan(self.p, self.centroids_p[k, :])

                # elif self.minkowski:

                tmp_distance = self.rho * tmp_distance_y + self.xi * tmp_distance_p
                distances.append(tmp_distance)

            distances = np.asarray(distances)
            self.clusters_labels = np.argmin(distances, axis=0)

            for k in set(self.clusters_labels):
                self.output_y[k] = self.y[np.where(self.clusters_labels == k)[0], :]
                self.output_p[k] = self.p[np.where(self.clusters_labels == k)[0], :]

            # Test whether all the new centroids are coincide with previous ones
            previous_centroids_y = deepcopy(self.centroids_y)
            previous_centroids_p = deepcopy(self.centroids_p)

            for k in set(self.clusters_labels):
                self.centroids_y[k, :] = np.mean(self.output_y[k], axis=0)
                self.centroids_p[k, :] = np.mean(self.output_p[k], axis=0)

            if np.all(self.centroids_y == previous_centroids_y) and np.all(self.centroids_p == previous_centroids_p):
                f_iter = False
            else:
                c_iter += 1

            if c_iter >= self.max_iterations:
                f_iter = False

        return self.clusters_labels


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--Name', type=str, default='--',
                        help='Name of the Experiment')

    parser.add_argument('--Run', type=int, default=0,
                        help='Whether to run the program or to evaluate the results')

    parser.add_argument('--Rho', type=float, default=1,
                        help='Feature coefficient during the clustering')

    parser.add_argument('--Xi', type=float, default=1,
                        help='Networks coefficient during the clustering')

    parser.add_argument('--N_clusters', type=int, default=5,
                        help='Number of clusters')

    parser.add_argument('--With_noise', type=int, default=0,
                        help='With noisy features or without')

    parser.add_argument('--PreProcessing', type=str, default='z-m',
                        help='string determining which pre processing method should be app_lied.'
                             'The first letter determines Y pre processing and the third determines P pre processing. '
                             'Separated with "-".')

    parser.add_argument('--Setting', type=str, default='all',)

    parser.add_argument('--Kmeans_pp', type=int, default=1,
                        help='Initialization method: If it is set to one K-Means Plus Plus will be applied, '
                             'otherwise, seeds will be instantiate randomly.')

    parser.add_argument('--Max_iterations', type=int, default=1000,
                        help='Maximum number of iteration')

    parser.add_argument('--Euclidean', type=int, default=0,
                        help='If it is set to one the Euclidean distance will be applied'
                             ' to measure the distance between centroids and data points')

    parser.add_argument('--Cosine', type=int, default=0,
                        help='If it is set to one the Cosine distance will be applied'
                             ' to measure the distances between centroids and data points')

    parser.add_argument('--Manhattan', type=int, default=1,
                        help='If it is set to one the Manhattan distance will be applied'
                             ' to measure the distances between centroids and data points')

    args = parser.parse_args()

    name, run, rho, xi, n_clusters, with_noise, pp, setting_,\
    max_iterations, kmean_pp, euclidean, cosine, manhattan = args_parser(args)

    data_name = name.split('(')[0]

    if manhattan == 1:
        euclidean = 0
        cosine = 0
    elif cosine == 1:
        euclidean = 0
        manhattan = 0
    elif euclidean == 1:
        cosine = 0
        manhattan = 0
    else:
        f = True
        print("Wrong distance function assignment.")
        assert f is True

    if with_noise == 1:
        data_name = data_name + "-N"

    type_of_data = name.split('(')[0][-1]

    start = time.time()

    if run == 1:

        with open(os.path.join('../data', name + ".pickle"), 'rb') as fp:
            DATA = pickle.load(fp)

        print("run:", name, run, rho, xi, n_clusters, with_noise, pp, "\n",
              setting_, type_of_data, 'cosine:', cosine, "\n",
              'euclidean:', euclidean, "Manhattan:", manhattan, "\n",
              'Kmeans plus plus:', kmean_pp)

        def apply_alg(data_type, with_noise):
            # Global initialization
            out_ms = {}

            if setting_ != 'all':

                for setting, repeats in DATA.items():

                    if str(setting) == setting_:

                        print("setting:", setting, )

                        out_ms[setting] = {}

                        for repeat, matrices in repeats.items():

                            print("repeat:", repeat)

                            GT = matrices['GT']
                            y = matrices['Y']
                            p = matrices['P']
                            y_n = matrices['Yn']
                            n, v = y.shape

                            p, p_sum_sim, p_ave_sim, p_u, p_u_sum_sim, pu_ave_sim, p_m, p_m_sum_sim, \
                                p_m_ave_sim, p_l, p_l_sum_sim, p_l_ave_sim = pt.preprocess_p(p=p)

                            # Quantitative case
                            if type_of_data == 'Q' or name.split('(')[-1] == 'r':
                                _, _, y_z, _, y_rng, _, = pt.preprocess_y(y_in=y, data_type='Q')

                                if with_noise == 1:
                                    y_n, _, y_n_z, _, y_n_rng, _, = pt.preprocess_y(y_in=y_n, data_type='Q')

                            # Because there is no y_n in the case of categorical features.
                            if type_of_data == 'C':
                                enc = OneHotEncoder(sparse=False, )  # categories='auto')
                                y_onehot = enc.fit_transform(y)  # oneHot encoding

                                # for WITHOUT follow-up rescale y_onehot and for
                                # "WITH follow-up" y_onehot should be rep_laced with Y
                                y, _, y_z, _, y_rng, _, = pt.preprocess_y(y_in=y_onehot, data_type='C')  # y_onehot

                            if type_of_data == 'M':
                                v_q = int(np.ceil(v / 2))  # number of quantitative features -- Y[:, :v_q]
                                v_c = int(np.floor(v / 2))  # number of categorical features  -- Y[:, v_q:]
                                _, _, y_z_q, _, y_rng_q, _, = pt.preprocess_y(y_in=y[:, :v_q], data_type='Q')
                                enc = OneHotEncoder(sparse=False, )  # categories='auto', )
                                y_onehot = enc.fit_transform(y[:, v_q:])  # oneHot encoding

                                # for WITHOUT follow-up rescale y_onehot and for
                                # "WITH follow-up" y_onehot should be rep_laced with Y
                                _, _, y_z_c, _, y_rng_c, _, = pt.preprocess_y(y_in=y[:, v_q:], data_type='C')

                                y = np.concatenate([y[:, :v_q], y_onehot], axis=1)
                                y_rng = np.concatenate([y_rng_q, y_rng_c], axis=1)
                                y_z = np.concatenate([y_z_q, y_z_c], axis=1)

                                if with_noise == 1:
                                    v_q = int(np.ceil(v / 2))  # number of quantitative features -- Y[:, :v_q]
                                    v_c = int(np.floor(v / 2))  # number of categorical features  -- Y[:, v_q:]
                                    v_qn = (v_q + v_c)  # the column index of which noise model1 starts

                                    _, _, y_n_z_q, _, y_n_rng_q, _, = pt.preprocess_y(y_in=y_n[:, :v_q], data_type='Q')

                                    enc = OneHotEncoder(sparse=False, )  # categories='auto',)
                                    y_n_onehot = enc.fit_transform(y_n[:, v_q:v_qn])  # oneHot encoding

                                    # for WITHOUT follow-up rescale y_n_oneHot and for
                                    # "WITH follow-up" y_n_oneHot should be rep_laced with Y
                                    y_n_c, _, y_n_z_c, _, y_n_rng_c, _, = pt.preprocess_y(y_in=y_n[:, v_q:v_qn],
                                                                                          data_type='C')  # y_n_oneHot

                                    y_ = np.concatenate([y_n[:, :v_q], y_n_onehot], axis=1)
                                    y_rng = np.concatenate([y_n_rng_q, y_n_rng_c], axis=1)
                                    y_z = np.concatenate([y_n_z_q, y_n_z_c], axis=1)

                                    _, _, y_n_z_, _, y_n_rng_, _, = pt.preprocess_y(y_in=y_n[:, v_qn:], data_type='Q')
                                    y_n_ = np.concatenate([y_, y_n[:, v_qn:]], axis=1)
                                    y_n_rng = np.concatenate([y_rng, y_n_rng_], axis=1)
                                    y_n_z = np.concatenate([y_z, y_n_z_], axis=1)

                            # Pre-processing - Without Noise
                            if data_type == "NP".lower() and with_noise == 0:
                                print("NP")
                                tmp_ms = EkifenMin(y=y, p=p, rho=rho, xi=xi,
                                                   n_clusters=n_clusters,
                                                   kmean_pp=kmean_pp,
                                                   euclidean=euclidean,
                                                   cosine=cosine,
                                                   manhattan=manhattan,
                                                   max_iteration=max_iterations).apply_ekifen()

                            elif data_type == "z-u".lower() and with_noise == 0:
                                print("z-u")
                                tmp_ms = EkifenMin(y=y_z, p=p_u, rho=rho, xi=xi,
                                                   n_clusters=n_clusters,
                                                   kmean_pp=kmean_pp,
                                                   euclidean=euclidean,
                                                   cosine=cosine,
                                                   manhattan=manhattan,
                                                   max_iteration=max_iterations).apply_ekifen()

                            elif data_type == "z-m".lower() and with_noise == 0:

                                tmp_ms = EkifenMin(y=y_z, p=p_m, rho=rho, xi=xi,
                                                   n_clusters=n_clusters,
                                                   kmean_pp=kmean_pp,
                                                   euclidean=euclidean,
                                                   cosine=cosine,
                                                   manhattan=manhattan,
                                                   max_iteration=max_iterations).apply_ekifen()

                            elif data_type == "z-l".lower() and with_noise == 0:
                                tmp_ms = EkifenMin(y=y_z, p=p_l, rho=rho, xi=xi,
                                                   n_clusters=n_clusters,
                                                   kmean_pp=kmean_pp,
                                                   euclidean=euclidean,
                                                   cosine=cosine,
                                                   manhattan=manhattan,
                                                   max_iteration=max_iterations).apply_ekifen()

                            elif data_type == "rng-u".lower() and with_noise == 0:
                                tmp_ms = EkifenMin(y=y_rng, p=p_u, rho=rho, xi=xi,
                                                   n_clusters=n_clusters,
                                                   kmean_pp=kmean_pp,
                                                   euclidean=euclidean,
                                                   cosine=cosine,
                                                   manhattan=manhattan,
                                                   max_iteration=max_iterations).apply_ekifen()

                            elif data_type == "rng-m".lower() and with_noise == 0:
                                tmp_ms = EkifenMin(y=y_rng, p=p_m, rho=rho, xi=xi,
                                                   n_clusters=n_clusters,
                                                   kmean_pp=kmean_pp,
                                                   euclidean=euclidean,
                                                   cosine=cosine,
                                                   manhattan=manhattan,
                                                   max_iteration=max_iterations).apply_ekifen()

                            elif data_type == "rng-l".lower() and with_noise == 0:
                                tmp_ms = EkifenMin(y=y_rng, p=p_l, rho=rho, xi=xi,
                                                   n_clusters=n_clusters,
                                                   kmean_pp=kmean_pp,
                                                   euclidean=euclidean,
                                                   cosine=cosine,
                                                   manhattan=manhattan,
                                                   max_iteration=max_iterations).apply_ekifen()

                            # Pre-processing - With Noise
                            if data_type == "NP".lower() and with_noise == 1:
                                tmp_ms = EkifenMin(y=y_n, p=p, rho=rho, xi=xi,
                                                   n_clusters=n_clusters,
                                                   kmean_pp=kmean_pp,
                                                   euclidean=euclidean,
                                                   cosine=cosine,
                                                   manhattan=manhattan,
                                                   max_iteration=max_iterations).apply_ekifen()

                            elif data_type == "z-u".lower() and with_noise == 1:
                                tmp_ms = EkifenMin(y=y_n_z, p=p_u, rho=rho, xi=xi,
                                                   n_clusters=n_clusters,
                                                   kmean_pp=kmean_pp,
                                                   euclidean=euclidean,
                                                   cosine=cosine,
                                                   manhattan=manhattan,
                                                   max_iteration=max_iterations).apply_ekifen()

                            elif data_type == "z-m".lower() and with_noise == 1:
                                tmp_ms = EkifenMin(y=y_n_z, p=p_m, rho=rho, xi=xi,
                                                   n_clusters=n_clusters,
                                                   kmean_pp=kmean_pp,
                                                   euclidean=euclidean,
                                                   cosine=cosine,
                                                   manhattan=manhattan,
                                                   max_iteration=max_iterations).apply_ekifen()

                            elif data_type == "z-l".lower() and with_noise == 1:
                                tmp_ms = EkifenMin(y=y_n_z, p=p_l, rho=rho, xi=xi,
                                                   n_clusters=n_clusters,
                                                   kmean_pp=kmean_pp,
                                                   euclidean=euclidean,
                                                   cosine=cosine,
                                                   manhattan=manhattan,
                                                   max_iteration=max_iterations).apply_ekifen()

                            elif data_type == "rng-u".lower() and with_noise == 1:
                                tmp_ms = EkifenMin(y=y_n_rng, p=p_u, rho=rho, xi=xi,
                                                   n_clusters=n_clusters,
                                                   kmean_pp=kmean_pp,
                                                   euclidean=euclidean,
                                                   cosine=cosine,
                                                   manhattan=manhattan,
                                                   max_iteration=max_iterations).apply_ekifen()

                            elif data_type == "rng-m".lower() and with_noise == 1:
                                tmp_ms = EkifenMin(y=y_n_rng, p=p_m, rho=rho, xi=xi,
                                                   n_clusters=n_clusters,
                                                   kmean_pp=kmean_pp,
                                                   euclidean=euclidean,
                                                   cosine=cosine,
                                                   manhattan=manhattan,
                                                   max_iteration=max_iterations).apply_ekifen()

                            elif data_type == "rng-l".lower() and with_noise == 1:
                                tmp_ms = EkifenMin(y=y_n_rng, p=p_l, rho=rho, xi=xi,
                                                   n_clusters=n_clusters,
                                                   kmean_pp=kmean_pp,
                                                   euclidean=euclidean,
                                                   cosine=cosine,
                                                   manhattan=manhattan,
                                                   max_iteration=max_iterations).apply_ekifen()

                            elif data_type == "rng-rng".lower():
                                print("rng-rng")
                                p, _, p_z, _, p_rng, _, = pt.preprocess_y(y_in=p, data_type='Q')
                                tmp_ms = EkifenMin(y=y_rng, p=p_rng, rho=rho, xi=xi,
                                                   n_clusters=n_clusters,
                                                   kmean_pp=kmean_pp,
                                                   euclidean=euclidean,
                                                   cosine=cosine,
                                                   manhattan=manhattan,
                                                   max_iteration=max_iterations).apply_ekifen()

                            elif data_type == "z-z".lower():
                                print("z-z")
                                p, _, p_z, _, p_rng, _, = pt.preprocess_y(y_in=p, data_type='Q')
                                tmp_ms = EkifenMin(y=y_z, p=p_z, rho=rho, xi=xi,
                                                   n_clusters=n_clusters,
                                                   kmean_pp=kmean_pp,
                                                   euclidean=euclidean,
                                                   cosine=cosine,
                                                   manhattan=manhattan,
                                                   max_iteration=max_iterations).apply_ekifen()

                            out_ms[setting][repeat] = tmp_ms

                print("Algorithm is app_lied on the entire data set!")
                print("setting:", setting_)

            if setting_ == 'all':

                for setting, repeats in DATA.items():

                    print("setting:", setting, )

                    out_ms[setting] = {}

                    for repeat, matrices in repeats.items():

                        print("repeat:", repeat)

                        GT = matrices['GT']
                        y = matrices['Y']
                        p = matrices['P']
                        y_n = matrices['Yn']
                        n, v = y.shape

                        p, p_sum_sim, p_ave_sim, p_u, p_u_sum_sim, pu_ave_sim, p_m, p_m_sum_sim, \
                            p_m_ave_sim, p_l, p_l_sum_sim, p_l_ave_sim = pt.preprocess_p(p=p)

                        # To preprocess networks as feature
                        _, _, p_z, _, p_rng, _, = pt.preprocess_y(y_in=p, data_type='Q')

                        # Quantitative case
                        if type_of_data == 'Q' or name.split('(')[-1] == 'r':
                            _, _, y_z, _, y_rng, _, = pt.preprocess_y(y_in=y, data_type='Q')

                            if with_noise == 1:
                                y_n, _, y_n_z, _, y_n_rng, _, = pt.preprocess_y(y_in=y_n, data_type='Q')

                        # Because there is no y_n in the case of categorical features.
                        if type_of_data == 'C':
                            enc = OneHotEncoder(sparse=False, )  # categories='auto')
                            y_onehot = enc.fit_transform(y)  # oneHot encoding

                            # for WITHOUT follow-up rescale y_onehot and for
                            # "WITH follow-up" y_onehot should be rep_laced with Y
                            y, _, y_z, _, y_rng, _, = pt.preprocess_y(y_in=y_onehot, data_type='C')  # y_onehot

                        if type_of_data == 'M':
                            v_q = int(np.ceil(v / 2))  # number of quantitative features -- Y[:, :v_q]
                            v_c = int(np.floor(v / 2))  # number of categorical features  -- Y[:, v_q:]
                            _, _, y_z_q, _, y_rng_q, _, = pt.preprocess_y(y_in=y[:, :v_q], data_type='Q')
                            enc = OneHotEncoder(sparse=False, )  # categories='auto', )
                            y_onehot = enc.fit_transform(y[:, v_q:])  # oneHot encoding

                            # for WITHOUT follow-up rescale y_onehot and for
                            # "WITH follow-up" y_onehot should be rep_laced with Y
                            _, _, y_z_c, _, y_rng_c, _, = pt.preprocess_y(y_in=y[:, v_q:],
                                                                          data_type='C')  # y_onehot

                            y = np.concatenate([y[:, :v_q], y_onehot], axis=1)
                            y_rng = np.concatenate([y_rng_q, y_rng_c], axis=1)
                            y_z = np.concatenate([y_z_q, y_z_c], axis=1)

                            if with_noise == 1:
                                v_q = int(np.ceil(v / 2))  # number of quantitative features -- Y[:, :v_q]
                                v_c = int(np.floor(v / 2))  # number of categorical features  -- Y[:, v_q:]
                                v_qn = (v_q + v_c)  # the column index of which noise model1 starts

                                _, _, y_n_z_q, _, y_n_rng_q, _, = pt.preprocess_y(y_in=y_n[:, :v_q], data_type='Q')

                                enc = OneHotEncoder(sparse=False, )  # categories='auto',)
                                y_n_onehot = enc.fit_transform(y_n[:, v_q:v_qn])  # oneHot encoding

                                # for WITHOUT follow-up rescale y_n_oneHot and for
                                # "WITH follow-up" y_n_oneHot should be rep_laced with Y
                                y_n_c, _, y_n_z_c, _, y_n_rng_c, _, = pt.preprocess_y(y_in=y_n[:, v_q:v_qn],
                                                                                      data_type='C')  # y_n_oneHot

                                y_ = np.concatenate([y_n[:, :v_q], y_n_onehot], axis=1)
                                y_rng = np.concatenate([y_n_rng_q, y_n_rng_c], axis=1)
                                y_z = np.concatenate([y_n_z_q, y_n_z_c], axis=1)

                                _, _, y_n_z_, _, y_n_rng_, _, = pt.preprocess_y(y_in=y_n[:, v_qn:], data_type='Q')
                                y_n_ = np.concatenate([y_, y_n[:, v_qn:]], axis=1)
                                y_n_rng = np.concatenate([y_rng, y_n_rng_], axis=1)
                                y_n_z = np.concatenate([y_z, y_n_z_], axis=1)

                        # Pre-processing - Without Noise
                        # Pre-processing - Without Noise
                        if data_type == "NP".lower() and with_noise == 0:
                            print("NP")
                            tmp_ms = EkifenMin(y=y, p=p, rho=rho, xi=xi,
                                               n_clusters=n_clusters,
                                               kmean_pp=kmean_pp,
                                               euclidean=euclidean,
                                               cosine=cosine,
                                               manhattan=manhattan,
                                               max_iteration=max_iterations).apply_ekifen()

                        elif data_type == "z-NP".lower() and with_noise == 0:
                            tmp_ms = EkifenMin(y=y_z, p=p, rho=rho, xi=xi,
                                               n_clusters=n_clusters,
                                               kmean_pp=kmean_pp,
                                               euclidean=euclidean,
                                               cosine=cosine,
                                               manhattan=manhattan,
                                               max_iteration=max_iterations).apply_ekifen()

                        elif data_type == "rng-NP".lower() and with_noise == 0:
                            tmp_ms = EkifenMin(y=y_rng, p=p, rho=rho, xi=xi,
                                               n_clusters=n_clusters,
                                               kmean_pp=kmean_pp,
                                               euclidean=euclidean,
                                               cosine=cosine,
                                               manhattan=manhattan,
                                               max_iteration=max_iterations).apply_ekifen()

                        elif data_type == "z-u".lower() and with_noise == 0:
                            print("z-u")
                            tmp_ms = EkifenMin(y=y_z, p=p_u, rho=rho, xi=xi,
                                               n_clusters=n_clusters,
                                               kmean_pp=kmean_pp,
                                               euclidean=euclidean,
                                               cosine=cosine,
                                               manhattan=manhattan,
                                               max_iteration=max_iterations).apply_ekifen()

                        elif data_type == "z-m".lower() and with_noise == 0:
                            tmp_ms = EkifenMin(y=y_z, p=p_m, rho=rho, xi=xi,
                                               n_clusters=n_clusters,
                                               kmean_pp=kmean_pp,
                                               euclidean=euclidean,
                                               cosine=cosine,
                                               manhattan=manhattan,
                                               max_iteration=max_iterations).apply_ekifen()

                        elif data_type == "z-l".lower() and with_noise == 0:
                            tmp_ms = EkifenMin(y=y_z, p=p_l, rho=rho, xi=xi,
                                               n_clusters=n_clusters,
                                               kmean_pp=kmean_pp,
                                               euclidean=euclidean,
                                               cosine=cosine,
                                               manhattan=manhattan,
                                               max_iteration=max_iterations).apply_ekifen()

                        elif data_type == "rng-u".lower() and with_noise == 0:
                            tmp_ms = EkifenMin(y=y_rng, p=p_u, rho=rho, xi=xi,
                                               n_clusters=n_clusters,
                                               kmean_pp=kmean_pp,
                                               euclidean=euclidean,
                                               cosine=cosine,
                                               manhattan=manhattan,
                                               max_iteration=max_iterations).apply_ekifen()

                        elif data_type == "rng-m".lower() and with_noise == 0:
                            tmp_ms = EkifenMin(y=y_rng, p=p_m, rho=rho, xi=xi,
                                               n_clusters=n_clusters,
                                               kmean_pp=kmean_pp,
                                               euclidean=euclidean,
                                               cosine=cosine,
                                               manhattan=manhattan,
                                               max_iteration=max_iterations).apply_ekifen()

                        elif data_type == "rng-l".lower() and with_noise == 0:
                            tmp_ms = EkifenMin(y=y_rng, p=p_l, rho=rho, xi=xi,
                                               n_clusters=n_clusters,
                                               kmean_pp=kmean_pp,
                                               euclidean=euclidean,
                                               cosine=cosine,
                                               manhattan=manhattan,
                                               max_iteration=max_iterations).apply_ekifen()

                        elif data_type == "rng-rng".lower() and with_noise == 0:
                            tmp_ms = EkifenMin(y=y_rng, p=p_rng, rho=rho, xi=xi,
                                               n_clusters=n_clusters,
                                               kmean_pp=kmean_pp,
                                               euclidean=euclidean,
                                               cosine=cosine,
                                               manhattan=manhattan,
                                               max_iteration=max_iterations).apply_ekifen()

                        elif data_type == "z-z".lower() and with_noise == 0:
                            tmp_ms = EkifenMin(y=y_z, p=p_z, rho=rho, xi=xi,
                                               n_clusters=n_clusters,
                                               kmean_pp=kmean_pp,
                                               euclidean=euclidean,
                                               cosine=cosine,
                                               manhattan=manhattan,
                                               max_iteration=max_iterations).apply_ekifen()

                        # Pre-processing - With Noise
                        if data_type == "NP".lower() and with_noise == 1:
                            tmp_ms = EkifenMin(y=y_n, p=p, rho=rho, xi=xi,
                                               n_clusters=n_clusters,
                                               kmean_pp=kmean_pp,
                                               euclidean=euclidean,
                                               cosine=cosine,
                                               manhattan=manhattan,
                                               max_iteration=max_iterations).apply_ekifen()

                        elif data_type == "z-u".lower() and with_noise == 1:
                            tmp_ms = EkifenMin(y=y_n_z, p=p_u, rho=rho, xi=xi,
                                               n_clusters=n_clusters,
                                               kmean_pp=kmean_pp,
                                               euclidean=euclidean,
                                               cosine=cosine,
                                               manhattan=manhattan,
                                               max_iteration=max_iterations).apply_ekifen()

                        elif data_type == "z-m".lower() and with_noise == 1:
                            tmp_ms = EkifenMin(y=y_n_z, p=p_m, rho=rho, xi=xi,
                                               n_clusters=n_clusters,
                                               kmean_pp=kmean_pp,
                                               euclidean=euclidean,
                                               cosine=cosine,
                                               manhattan=manhattan,
                                               max_iteration=max_iterations).apply_ekifen()

                        elif data_type == "z-l".lower() and with_noise == 1:
                            tmp_ms = EkifenMin(y=y_n_z, p=p_l, rho=rho, xi=xi,
                                               n_clusters=n_clusters,
                                               kmean_pp=kmean_pp,
                                               euclidean=euclidean,
                                               cosine=cosine,
                                               manhattan=manhattan,
                                               max_iteration=max_iterations).apply_ekifen()

                        elif data_type == "rng-u".lower() and with_noise == 1:
                            tmp_ms = EkifenMin(y=y_n_rng, p=p_u, rho=rho, xi=xi,
                                               n_clusters=n_clusters,
                                               kmean_pp=kmean_pp,
                                               euclidean=euclidean,
                                               cosine=cosine,
                                               manhattan=manhattan,
                                               max_iteration=max_iterations).apply_ekifen()

                        elif data_type == "rng-m".lower() and with_noise == 1:
                            tmp_ms = EkifenMin(y=y_n_rng, p=p_m, rho=rho, xi=xi,
                                               n_clusters=n_clusters,
                                               kmean_pp=kmean_pp,
                                               euclidean=euclidean,
                                               cosine=cosine,
                                               manhattan=manhattan,
                                               max_iteration=max_iterations).apply_ekifen()

                        elif data_type == "rng-l".lower() and with_noise == 1:
                            tmp_ms = EkifenMin(y=y_n_rng, p=p_l, rho=rho, xi=xi,
                                               n_clusters=n_clusters,
                                               kmean_pp=kmean_pp,
                                               euclidean=euclidean,
                                               cosine=cosine,
                                               manhattan=manhattan,
                                               max_iteration=max_iterations).apply_ekifen()

                        elif data_type == "rng-rng".lower():
                            print("rng-rng")
                            tmp_ms = EkifenMin(y=y_rng, p=p_rng, rho=rho, xi=xi,
                                               n_clusters=n_clusters,
                                               kmean_pp=kmean_pp,
                                               euclidean=euclidean,
                                               cosine=cosine,
                                               manhattan=manhattan,
                                               max_iteration=max_iterations).apply_ekifen()

                        elif data_type == "z-z".lower():
                            print("z-z")
                            tmp_ms = EkifenMin(y=y_z, p=p_z, rho=rho, xi=xi,
                                               n_clusters=n_clusters,
                                               kmean_pp=kmean_pp,
                                               euclidean=euclidean,
                                               cosine=cosine,
                                               manhattan=manhattan,
                                               max_iteration=max_iterations).apply_ekifen()

                        out_ms[setting][repeat] = tmp_ms

                print("Algorithm is app_lied on the entire data set!")

            return out_ms

        out_ms = apply_alg(data_type=pp.lower(), with_noise=with_noise)

        end = time.time()

        print("Time:", end - start)

        if with_noise == 1:
            name = name + '-N'

        if euclidean:
            if setting_ != 'all':
                with open(os.path.join('../data', "EKiFeNe" + name + "-" + pp + "-" + setting_ + "-" +
                                                  str(n_clusters) + ".pickle"), 'wb') as fp:
                    pickle.dump(out_ms, fp)

            if setting_ == 'all':
                with open(os.path.join('../data', "EKiFeNe" + name + "-" + pp + "-" +
                                                  str(n_clusters) + ".pickle"), 'wb') as fp:
                    pickle.dump(out_ms, fp)

        elif cosine:
            if setting_ != 'all':
                with open(os.path.join('../data', "EKiFeNc" + name + "-" + pp + "-" + setting_ + "-" +
                                                  str(n_clusters) + ".pickle"), 'wb') as fp:
                    pickle.dump(out_ms, fp)

            if setting_ == 'all':
                with open(os.path.join('../data', "EKiFeNc" + name + "-" + pp + "-" +
                                                  str(n_clusters) + ".pickle"), 'wb') as fp:
                    pickle.dump(out_ms, fp)

        elif manhattan:
            if setting_ != 'all':
                with open(os.path.join('../data', "EKiFeNm" + name + "-" + pp + "-" + setting_ + "-" +
                                                  str(n_clusters) + ".pickle"), 'wb') as fp:
                    pickle.dump(out_ms, fp)

            if setting_ == 'all':
                with open(os.path.join('../data', "EKiFeNm" + name + "-" + pp + "-" +
                                                  str(n_clusters) + ".pickle"), 'wb') as fp:
                    pickle.dump(out_ms, fp)

        print("Results are saved!")

        print(" ")
        print("\t", "  p", "  q", " a/e", "\t", "  ARI     ", "  NMI", )
        print(" \t", " \t", " \t", " Ave", " std", " Ave", " std")
        for setting, results in out_ms.items():
            ARI, NMI = [], []
            for repeat, result in results.items():
                lp = result
                if not name.split('(')[-1] == 'r':
                    gt, gti = pt.flat_ground_truth(DATA[setting][repeat]['GT'])
                else:
                    gt = DATA[setting][repeat]['GT']
                    gt = [int(ii) for ii in gt]
                ARI.append(metrics.adjusted_rand_score(gt, lp))
                NMI.append(metrics.adjusted_mutual_info_score(gt, lp))

            ari_ave = np.mean(np.asarray(ARI), axis=0)
            ari_std = np.std(np.asarray(ARI), axis=0)
            nmi_ave = np.mean(np.asarray(NMI), axis=0)
            nmi_std = np.std(np.asarray(NMI), axis=0)
            print("setting:", setting, "%.3f" % ari_ave, "%.3f" % ari_std, "%.3f" % nmi_ave,
                  "%.3f" % nmi_std)

        print(" ")
        print("\t", "  p", "  q", " a/e   ", "precision,", 'recall', '  f-score')
        print(" \t", " \t", " \t", "Ave", " std", " Ave", " std", " Ave", " std")
        for setting, results in out_ms.items():
            precision, recall, fscore = [], [], []
            for repeat, result in results.items():
                lp = result
                if not name.split('(')[-1] == 'r':
                    gt, gti = pt.flat_ground_truth(DATA[setting][repeat]['GT'])
                else:
                    gt = DATA[setting][repeat]['GT']
                    gt = [int(ii) for ii in gt]
                tmp = metrics.precision_recall_fscore_support(gt, lp, average='weighted')

                precision.append(tmp[0])
                recall.append(tmp[1])
                fscore.append(tmp[2])

            precision_ave = np.mean(np.asarray(precision), axis=0)
            precision_std = np.std(np.asarray(precision), axis=0)

            recall_ave = np.mean(np.asarray(recall), axis=0)
            recall_std = np.std(np.asarray(recall), axis=0)

            fscore_ave = np.mean(np.asarray(fscore), axis=0)
            fscore_std = np.std(np.asarray(fscore), axis=0)

            print("setting:", setting, "%.3f" % precision_ave, "%.3f" % precision_std,
                  "%.3f" % recall_ave, "%.3f" % recall_std,
                  "%.3f" % fscore_ave, "%.3f" % fscore_std,
                  )

        print(" ")
        print(" Number of detected clusters")
        print(" \t", " \t", "   Ave", "  std", )
        for setting, results in out_ms.items():
            num_cluster = []
            for repeat, result in results.items():
                num_cluster.append(int(len(set(result))))

            ave_num_clust = np.mean(np.asarray(num_cluster), axis=0)
            std_num_clust = np.std(np.asarray(num_cluster), axis=0)
            print("Number of Clusters:", "%.3f" % ave_num_clust, "%.3f" % std_num_clust)

    if run == 0:

        print(" \t", " \t", "name:", name)

        with open(os.path.join('../data', name + ".pickle"), 'rb') as fp:
            DATA = pickle.load(fp)

        if with_noise == 1:
            name = name + '-N'

        if euclidean == 1:
            if setting_ != 'all':
                with open(os.path.join('../data', "EKiFeNe" + name + "-" + pp + "-" + setting_ + "-" +
                                                  str(n_clusters) + ".pickle"), 'rb') as fp:
                    out_ms = pickle.load(fp)

            if setting_ == 'all':
                with open(os.path.join('../data', "EKiFeNe" + name + "-" + pp + "-" +
                                                  str(n_clusters) + ".pickle"), 'rb') as fp:
                    out_ms = pickle.load(fp)

        elif cosine == 1:
            if setting_ != 'all':
                with open(os.path.join('../data', "EKiFeNc" + name + "-" + pp + "-" + setting_ + "-" +
                                                  str(n_clusters) + ".pickle"), 'rb') as fp:
                    out_ms = pickle.load(fp)

            if setting_ == 'all':
                with open(os.path.join('../data', "EKiFeNc" + name + "-" + pp + "-" +
                                                  str(n_clusters) + ".pickle"), 'rb') as fp:
                    out_ms = pickle.load(fp)

        elif manhattan == 1:
            if setting_ != 'all':
                with open(os.path.join('../data', "EKiFeNm" + name + "-" + pp + "-" + setting_ + "-" +
                                                  str(n_clusters) + ".pickle"), 'rb') as fp:
                    out_ms = pickle.load(fp)

            if setting_ == 'all':
                with open(os.path.join('../data', "EKiFeNm" + name + "-" + pp + "-" +
                                                  str(n_clusters) + ".pickle"), 'rb') as fp:
                    out_ms = pickle.load(fp)

        print(" ")
        print("\t", "  p", "  q", " a/e", "\t", "  ARI     ", "  NMI", )
        print(" \t", " \t", " \t", " Ave", " std", " Ave", " std")

        for setting, results in out_ms.items():
            ARI, NMI = [], []
            for repeat, result in results.items():
                lp = result

                if not name.split('(')[-1] == 'r':
                    gt, gti = pt.flat_ground_truth(DATA[setting][repeat]['GT'])
                else:
                    gt = DATA[setting][repeat]['GT']

                ARI.append(metrics.adjusted_rand_score(gt, lp))
                NMI.append(metrics.adjusted_mutual_info_score(gt, lp))

            ari_ave = np.mean(np.asarray(ARI), axis=0)
            ari_std = np.std(np.asarray(ARI), axis=0)
            nmi_ave = np.mean(np.asarray(NMI), axis=0)
            nmi_std = np.std(np.asarray(NMI), axis=0)
            print("setting:", setting, "%.3f" % ari_ave, "%.3f" % ari_std, "%.3f" % nmi_ave,
                  "%.3f" % nmi_std)

        print(" ")
        print("\t", "  p", "  q", " a/e   ", "precision,", 'recall', '  f-score')
        print(" \t", " \t", " \t", "Ave", " std", " Ave", " std", " Ave", " std")
        for setting, results in out_ms.items():
            precision, recall, fscore = [], [], []
            for repeat, result in results.items():
                lp = result

                if not name.split('(')[-1] == 'r':
                    gt, gti = pt.flat_ground_truth(DATA[setting][repeat]['GT'])
                else:
                    gt = DATA[setting][repeat]['GT']

                tmp = metrics.precision_recall_fscore_support(gt, lp, average='weighted')

                precision.append(tmp[0])
                recall.append(tmp[1])
                fscore.append(tmp[2])

            precision_ave = np.mean(np.asarray(precision), axis=0)
            precision_std = np.std(np.asarray(precision), axis=0)

            recall_ave = np.mean(np.asarray(recall), axis=0)
            recall_std = np.std(np.asarray(recall), axis=0)

            fscore_ave = np.mean(np.asarray(fscore), axis=0)
            fscore_std = np.std(np.asarray(fscore), axis=0)

            print("setting:", setting, "%.3f" % precision_ave, "%.3f" % precision_std,
                  "%.3f" % recall_ave, "%.3f" % recall_std,
                  "%.3f" % fscore_ave, "%.3f" % fscore_std,
                  )

        print(" ")
        print(" Number of detected clusters")
        print(" \t", " \t", "   Ave", "  std", )
        for setting, results in out_ms.items():
            num_cluster = []
            for repeat, result in results.items():
                num_cluster.append(int(len(set(result))))

            ave_num_clust = np.mean(np.asarray(num_cluster), axis=0)
            std_num_clust = np.std(np.asarray(num_cluster), axis=0)
            print("Number of Clusters:", "%.3f" % ave_num_clust, "%.3f" % std_num_clust)

        # print("contingency tables")
        # for setting, results in out_ms.items():
        #     for repeat, result in results.items():
        #         lp, lpi = flat_cluster_results(result)
        #         gt, gti = flat_ground_truth(DATA[setting][repeat]['GT'])
        #         tmp_cont, _, _ = ev.sk_contingency(tmp_out_ms, tmp_out_gt,)  # N
        #         print("setting:", setting, repeat)
        #         print(tmp_cont)
        #         print(" ")









