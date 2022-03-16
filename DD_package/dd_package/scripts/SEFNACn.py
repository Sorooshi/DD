# -*- coding: utf-8 -*-

import os
import time
import pickle
import argparse
import warnings
import numpy as np
from sklearn import metrics
import processing_tools as pt
import similarity_generator as sg
from collections import OrderedDict
from sklearn.preprocessing import OneHotEncoder


warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True, linewidth=120, precision=2)

# np.random.seed(42)  # 3016222231 76(54)

current_seed = np.random.get_state()[1][0]
with open('granary.txt', 'a+') as o:
    o.write(str(current_seed)+'\n')


def args_parser(args):

    name = args.Name
    run = args.Run
    rho = args.Rho
    xi = args.Xi
    num_pivots = args.NumPivots
    range_pivots = args.RangePivots
    with_noise = args.With_noise
    pp = args.PreProcessing
    setting = args.Setting
    feature_embedding_is_inner = args.Feature_embedding_is_inner

    return name, run, rho, xi, num_pivots, range_pivots, with_noise, pp, setting, feature_embedding_is_inner


def compute_contribution(s, p, rho, xi, indices):

    cv_y = np.sum(np.power(np.mean(s[indices, :], axis=0), 2))  # Cv in the criterion
    cv_p = np.sum(np.power(np.mean(p[indices, :], axis=0), 2))  # Cj in the criterion
    contribution = rho * cv_y * len(indices) + xi * cv_p * len(indices)

    return contribution


def stage1(input_stage1, k):

    pivot = input_stage1['pivot']
    clustered_indices = input_stage1['clustered_indices']  # list; i.e. list of cluster's membership vectors

    output_stage1 = OrderedDict()
    output_stage1['clustered_indices'] = clustered_indices
    output_stage1['pivot'] = pivot
    output_stage1['k'] = k
    return output_stage1


def stage2(s, p, output_stage1, k, rho, xi):

    n, v = s.shape
    total_indices = list(range(n))
    output_stage2 = OrderedDict()

    pivot = output_stage1['pivot']
    clustered_indices = output_stage1['clustered_indices']  # all clustered indices except the pivot
    remained_indices = list(set(total_indices).difference(clustered_indices))
    remained_indices.remove(pivot)  # because pivot from stage1 has not been added to clustered indices yet!

    r = len(remained_indices)  # remained entries,

    cntr = np.zeros([r, 2])  # cluster Contribution
    index = 0

    for i in range(n):
        if i in remained_indices:  # and not (i in p_abnormalities):
            cntr[index, :] = (compute_contribution(s=s, p=p, rho=rho, xi=xi, indices=[pivot, i]), i)
            index += 1

    argmax = int(np.argmax(cntr[:, 0]))  # k-th cluster's contribution

    clustered_indices_k = list([pivot, int(cntr[argmax, -1])])  # list, membership vector of this cluster
    output_stage2['clustered_indices-' + str(k)] = clustered_indices_k
    output_stage2['clustered_indices'] = clustered_indices  # list of all clustered indices

    return output_stage2


def stage3(s, p, output_stage2, k, rho, xi):

    n, v = s.shape
    total_indices = list(range(n))
    num_elements = 2

    clustered_indices = output_stage2['clustered_indices']  # all clustered indices
    clustered_indices_k = output_stage2['clustered_indices-' + str(k)]  # pivot + argmax stage2

    cntr_old = compute_contribution(s=s, p=p, rho=rho, xi=xi, indices=clustered_indices_k)

    remained_indices = list(set(total_indices).difference(clustered_indices))

    for i in clustered_indices_k:
        remained_indices.remove(i)  # removing the pivot and the argamx of stage2 from the remaining list

    f_2 = True

    output_stage3 = OrderedDict()

    while f_2 and len(remained_indices) != 0:

        list_of_candidates = [clustered_indices_k + [i] for i in remained_indices if i not in clustered_indices_k]

        cntr = [compute_contribution(s=s, p=p, rho=rho, xi=xi, indices=indices) for indices in list_of_candidates]

        cntr = np.asarray(cntr)
        delta = float(np.subtract(float(np.max(cntr, axis=0)), cntr_old))
        argmax = int(np.argmax(cntr, axis=0))
        list_of_indices_to_append = list(set(list_of_candidates[argmax]).difference(clustered_indices_k))

        # k-th cluster's contribution
        cntr_old = float(np.max(cntr, axis=0))

        # print('R end while:', R, "K:", k)

        if delta > 0:
            num_elements += 1
            if num_elements == 3:
                clustered_indices += [l for l in clustered_indices_k]

            # list, update membership vector of k-th cluster
            clustered_indices_k += list_of_indices_to_append  # [ind for ind in list_of_indices_to_append]
            # list, update the total clustered indices
            clustered_indices += list_of_indices_to_append  # [ind for ind in list_of_indices_to_append]

            remained_indices = list(set(total_indices).difference(clustered_indices))
            output_stage3['bad_start'] = False
        else:
            f_2 = False

    if num_elements == 2:  # if I'm not mistaken it should be only two elements not three!

        # in the "clustered_indices" list' pivot and argmax of stage2 are not included
        remained_indices = list(set(total_indices).difference(clustered_indices))

        # it was a bad start and there are enough elements (>=3) for clustering. i.e pivot + argmax2 +
        # at least three remaining elements (one or two extra elements are also acceptable)
        if len(remained_indices) > 6:
            # print("more than 3 el.")
            # a list of possible candidate for pivots in order to avoid reselection of the same elements
            candidates = remained_indices

            for i in clustered_indices_k:
                candidates.remove(i)  # removing pivot and argmax stage2 (check the correctness)

            output_stage3['pivot'] = np.random.choice(candidates)
            output_stage3['bad_start'] = True  # 99% it includes just two elements!
            output_stage3['continue'] = True
            output_stage3['clustered_indices-' + str(k)] = clustered_indices_k
            output_stage3['clustered_indices'] = clustered_indices

        # it was a bad start but there ARE NOT enough elements (<3) for clustering,
        # therefore, put all the remaining into a cluster
        else:
            output_stage3['continue'] = False
            output_stage3['bad_start'] = False
            clustered_indices_k = remained_indices
            clustered_indices += [i for i in remained_indices]
            output_stage3['clustered_indices-' + str(k)] = clustered_indices_k
            output_stage3['clustered_indices'] = clustered_indices

    else:  # it was NOT A BAD start,
        remained_indices = list(set(total_indices).difference(clustered_indices))
        # print("HERE")

        if len(remained_indices) > 3:
            output_stage3['continue'] = True

            # contribution check i.e. the correctness of pivot:
            cntr_org = compute_contribution(s=s, p=p, rho=rho, xi=xi, indices=clustered_indices_k)

            tmp_pivot = clustered_indices_k.pop(0)  # removing pivot

            # without pivot
            cntr_norg = compute_contribution(s=s, p=p, rho=rho, xi=xi, indices=clustered_indices_k)

            if cntr_norg > cntr_org:  # wrong pivot was chosen

                output_stage3['clustered_indices-' + str(k)] = clustered_indices_k
                clustered_indices.remove(tmp_pivot)

                output_stage3['clustered_indices'] = clustered_indices
                remained_indices = list(set(total_indices).difference(clustered_indices))
                output_stage3['pivot'] = np.random.choice(remained_indices)  # pivot is not included

            else:  # correct pivot was chosen
                remained_indices = list(set(total_indices).difference(clustered_indices))

                clustered_indices_k.insert(0, tmp_pivot)
                output_stage3['clustered_indices-' + str(k)] = clustered_indices_k
                output_stage3['clustered_indices'] = clustered_indices  #
                output_stage3['pivot'] = np.random.choice(remained_indices)
        else:
            output_stage3['continue'] = False
            output_stage3['pivot'] = []
            clustered_indices_k += [j for j in remained_indices if j not in clustered_indices_k]
            clustered_indices += [jj for jj in remained_indices if jj not in clustered_indices]
            output_stage3['clustered_indices-' + str(k)] = clustered_indices_k
            output_stage3['clustered_indices'] = clustered_indices

    return output_stage3


def run_ANomalous_Cluster(pivot, y, p, rho, xi):

    # Initialization
    ry, _ = y.shape
    total_indices = set(range(ry))

    f_1 = True

    k = 0  # Number of Cluster

    input_stage1 = OrderedDict()
    input_stage1['pivot'] = pivot
    input_stage1['clustered_indices'] = []

    clustering_results = OrderedDict()  # to store the total clustering results where key is the k-th cluster__ str(k).

    counter = 0

    while f_1:

        output_stage1 = stage1(input_stage1=input_stage1, k=k)
        output_stage2 = stage2(s=y, p=p, output_stage1=output_stage1, k=k, rho=rho, xi=xi)
        output_stage3 = stage3(s=y, p=p, output_stage2=output_stage2, k=k, rho=rho, xi=xi)

        if output_stage3['continue'] is True and output_stage3['bad_start'] is False:

            counter = 0

            clustering_results[str(k)] = output_stage3['clustered_indices-' + str(k)]
            input_stage1['pivot'] = output_stage3['pivot']
            input_stage1['clustered_indices'] = output_stage3['clustered_indices']
            remained_indices = list(set(total_indices).difference(output_stage3['clustered_indices']))

            # there are just three elements remained, it is better to consider them as the extension of previous cluster
            if len(remained_indices) <= 3 and len(remained_indices) != 0:

                tmp_clustering_result_k = clustering_results[str(k)]

                for r in remained_indices:

                    if r not in tmp_clustering_result_k:
                        tmp_clustering_result_k.append(r)

                clustering_results[str(k)] += tmp_clustering_result_k

                f_1 = False

            k += 1

        elif output_stage3['continue'] is True and output_stage3['bad_start'] is True:
            counter += 1
            input_stage1['pivot'] = output_stage3['pivot']
            input_stage1['clustered_indices'] = output_stage3['clustered_indices']

            if counter >= 10:  # An extra protection in order to avoid stocking in local optima

                clustered_indices_k = output_stage3['clustered_indices-' + str(k)]  # a cluster of two abnormal elements
                # print("clustered_indices_k:", clustered_indices_k, len(clustered_indices_k))

                clustering_results[str(k)] = clustered_indices_k
                clustered_indices = output_stage3['clustered_indices']

                # print("clustered_indices:", clustered_indices, len(clustered_indices))

                # bcz, pivot and argmax2 were not added in stage3 of this case
                # (the if condition is added for extra check)
                clustered_indices += [i for i in clustered_indices_k if i not in clustered_indices]

                input_stage1['clustered_indices'] = clustered_indices

                counter = 0
                k += 1

        elif output_stage3['continue'] is False and output_stage3['bad_start'] is False:
            clustering_results[str(k)] = output_stage3['clustered_indices-' + str(k)]
            f_1 = False

    return clustering_results


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

    parser.add_argument('--NumPivots', type=int, default=1,
                        help='Number of pivots for initialization of clustering process')

    parser.add_argument('--RangePivots', type=int, default=200,
                        help='A Range to chose pivots for initialization of clustering process')

    parser.add_argument('--With_noise', type=int, default=0,
                        help='With noisy features or without')

    parser.add_argument('--PreProcessing', type=str, default='z-m',
                        help='string determining which pre processing method should be app_lied.'
                             'The first letter determines Y pre processing and the third determines P pre processing. '
                             'Separated with "-".')

    parser.add_argument('--Setting', type=str, default='all')

    parser.add_argument('--Feature_embedding_is_inner', type=int, default=1,
                        help='If one the embedding of features is done with inner product otherwise with kernel trick')

    args = parser.parse_args()

    name, run, rho, xi, num_pivots, range_pivots, with_noise, \
        pp, setting_, feature_embedding_is_inner = args_parser(args)

    data_name = name.split('(')[0]

    if with_noise == 1:
        data_name = data_name + "-N"

    type_of_data = name.split('(')[0][-1]

    start = time.time()

    if run == 1:

        with open(os.path.join('../data', name + ".pickle"), 'rb') as fp:
            DATA = pickle.load(fp)

        pivot = np.random.choice(range_pivots, num_pivots, replace=False)[0]
        print("pivot:", pivot)

        print("run:", name, run, rho, xi, num_pivots, range_pivots, with_noise,
              pp, setting_, feature_embedding_is_inner, type_of_data)

        def apply_ac(data_type, with_noise):

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
                                s_inn, s_aff = sg.generate_similarities(y=y_rng)

                            if type_of_data == 'M':
                                v_q = int(np.ceil(v / 2))  # number of quantitative features -- Y[:, :v_q]
                                v_c = int(np.floor(v / 2))  # number of categorical features  -- Y[:, v_q:]
                                _, _, y_z_q, _, y_rng_q, _, = pt.preprocess_y(y_in=y[:, :v_q], data_type='Q')
                                enc = OneHotEncoder(sparse=False, )  # categories='auto', )
                                y_onehot = enc.fit_transform(y[:, v_q:])  # oneHot encoding

                                # for WITHOUT follow-up rescale y_onehot and for
                                # "WITH follow-up" y_onehot should be rep_laced with Y
                                _, _, y_z_c, _, y_rng_c, _, = pt.preprocess_y(y_in=y[:, v_q:], data_type='C')  # y_onehot

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
                                tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y, p=p, rho=rho, xi=xi)

                            elif data_type == "z-u".lower() and with_noise == 0:
                                print("z-u")
                                tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_z, p=p_u, rho=rho,
                                                               xi=xi)

                            elif data_type == "z-m".lower() and with_noise == 0:
                                tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_z, p=p_m, rho=rho,
                                                               xi=xi)

                            elif data_type == "z-l".lower() and with_noise == 0:
                                tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_z, p=p_l, rho=rho,
                                                               xi=xi)

                            elif data_type == "rng-u".lower() and with_noise == 0:
                                tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_rng, p=p_u, rho=rho,
                                                               xi=xi)

                            elif data_type == "rng-m".lower() and with_noise == 0:
                                tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_rng, p=p_m, rho=rho,
                                                               xi=xi)

                            elif data_type == "rng-l".lower() and with_noise == 0:
                                tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_rng, p=p_l, rho=rho,
                                                               xi=xi)

                            # Pre-processing - With Noise
                            if data_type == "NP".lower() and with_noise == 1:
                                tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_n, p=p, rho=rho, xi=xi)

                            elif data_type == "z-u".lower() and with_noise == 1:
                                tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_n_z, p=p_u, rho=rho,
                                                               xi=xi)

                            elif data_type == "z-m".lower() and with_noise == 1:
                                tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_n_z, p=p_m, rho=rho,
                                                               xi=xi)

                            elif data_type == "z-l".lower() and with_noise == 1:
                                tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_n_z, p=p_l, rho=rho,
                                                               xi=xi)

                            elif data_type == "rng-u".lower() and with_noise == 1:
                                tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_n_rng, p=p_u, rho=rho,
                                                               xi=xi)

                            elif data_type == "rng-m".lower() and with_noise == 1:
                                tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_n_rng, p=p_m, rho=rho,
                                                               xi=xi)

                            elif data_type == "rng-l".lower() and with_noise == 1:
                                tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_n_rng, p=p_l, rho=rho,
                                                               xi=xi)

                            elif data_type == "rng-rng".lower():
                                print("rng-rng")
                                p, _, p_z, _, p_rng, _, = pt.preprocess_y(y_in=p, data_type='Q')
                                tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_rng, p=p_rng, rho=rho,
                                                               xi=xi)

                            elif data_type == "z-z".lower():
                                print("z-z")
                                p, _, p_z, _, p_rng, _, = pt.preprocess_y(y_in=p, data_type='Q')
                                tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_z, p=p_z, rho=rho,
                                                               xi=xi)

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

                        # to preprocess networks as features
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
                            s_inn, s_aff = sg.generate_similarities(y=y_rng)

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
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y, p=p, rho=rho, xi=xi)

                        elif data_type == "z-NP".lower() and with_noise == 0:
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_z, p=p, rho=rho, xi=xi)

                        elif data_type == "rng-NP".lower() and with_noise == 0:
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_rng, p=p, rho=rho, xi=xi)

                        elif data_type == "z-u".lower() and with_noise == 0:
                            print("z-u")
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_z, p=p_u, rho=rho,
                                                           xi=xi)

                        elif data_type == "z-m".lower() and with_noise == 0:
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_z, p=p_m, rho=rho,
                                                           xi=xi)

                        elif data_type == "z-l".lower() and with_noise == 0:
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_z, p=p_l, rho=rho,
                                                           xi=xi)

                        elif data_type == "rng-u".lower() and with_noise == 0:
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_rng, p=p_u, rho=rho,
                                                           xi=xi)

                        elif data_type == "rng-m".lower() and with_noise == 0:
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_rng, p=p_m, rho=rho,
                                                           xi=xi)

                        elif data_type == "rng-l".lower() and with_noise == 0:
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_rng, p=p_l, rho=rho,
                                                           xi=xi)

                        # Pre-processing - With Noise
                        if data_type == "NP".lower() and with_noise == 1:
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_n, p=p, rho=rho, xi=xi)

                        elif data_type == "z-u".lower() and with_noise == 1:
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_n_z, p=p_u, rho=rho,
                                                           xi=xi)

                        elif data_type == "z-m".lower() and with_noise == 1:
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_n_z, p=p_m, rho=rho,
                                                           xi=xi)

                        elif data_type == "z-l".lower() and with_noise == 1:
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_n_z, p=p_l, rho=rho,
                                                           xi=xi)

                        elif data_type == "rng-u".lower() and with_noise == 1:
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_n_rng, p=p_u, rho=rho,
                                                           xi=xi)

                        elif data_type == "rng-m".lower() and with_noise == 1:
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_n_rng, p=p_m, rho=rho,
                                                           xi=xi)

                        elif data_type == "rng-l".lower() and with_noise == 1:
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_n_rng, p=p_l, rho=rho,
                                                           xi=xi)

                        elif data_type == "rng-rng".lower():
                            print("rng-rng")
                            p, _, p_z, _, p_rng, _, = pt.preprocess_y(y_in=p, data_type='Q')
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_rng, p=p_rng, rho=rho,
                                                           xi=xi)

                        elif data_type == "z-z".lower():
                            print("z-z")
                            p, _, p_z, _, p_rng, _, = pt.preprocess_y(y_in=p, data_type='Q')
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, y=y_z, p=p_z, rho=rho,
                                                           xi=xi)

                        out_ms[setting][repeat] = tmp_ms

                print("Algorithm is app_lied on the entire data set!")

            return out_ms

        out_ms = apply_ac(data_type=pp.lower(), with_noise=with_noise)

        end = time.time()

        print("Time:", end - start)

        if with_noise == 1:
            name = name + '-N'

        if setting_ != 'all':
            with open(os.path.join('../data', "FNF_" + name + "-" + pp + "-" + setting_ + "-" +
                                              str(feature_embedding_is_inner) + ".pickle"), 'wb') as fp:
                pickle.dump(out_ms, fp)

        if setting_ == 'all':
            with open(os.path.join('../data', "FNF_" + name + "-" + pp + "-" +
                                              str(feature_embedding_is_inner) + ".pickle"), 'wb') as fp:
                pickle.dump(out_ms, fp)

        print("Results are saved!")

        print(" ")
        print("\t", "  p", "  q", " a/e", "\t", "  ARI     ", "  NMI", )
        print(" \t", " \t", " \t", " Ave", " std", " Ave", " std")
        for setting, results in out_ms.items():
            ARI, NMI = [], []
            for repeat, result in results.items():
                lp, lpi = pt.flat_cluster_results(result)
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
                lp, _ = pt.flat_cluster_results(result)
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
                num_cluster.append(int(len(result.keys())))

            ave_num_clust = np.mean(np.asarray(num_cluster), axis=0)
            std_num_clust = np.std(np.asarray(num_cluster), axis=0)
            print("Number of Clusters:", "%.3f" % ave_num_clust, "%.3f" % std_num_clust)

    if run == 0:

        print(" \t", " \t", "name:", name)

        with open(os.path.join('../data', name + ".pickle"), 'rb') as fp:
            DATA = pickle.load(fp)

        if with_noise == 1:
            name = name + '-N'

        if setting_ != 'all':

            with open(os.path.join('../data', "FNF_" + name + "-" + pp + "-" + setting_ + "-" +
                                              str(feature_embedding_is_inner) + ".pickle"), 'rb') as fp:
                out_ms = pickle.load(fp)

        if setting_ == 'all':
            with open(os.path.join('../data', "FNF_" + name + "-" + pp + "-" +
                                              str(feature_embedding_is_inner) + ".pickle"), 'rb') as fp:
                out_ms = pickle.load(fp)

        print(" ")
        print("\t", "  p", "  q", " a/e", "\t", "  ARI     ", "  NMI", )
        print(" \t", " \t", " \t", " Ave", " std", " Ave", " std")
        for setting, results in out_ms.items():
            ARI, NMI = [], []
            for repeat, result in results.items():
                lp, _ = pt.flat_cluster_results(result)

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
                lp, _ = pt.flat_cluster_results(result)

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
                num_cluster.append(int(len(result.keys())))

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

