import os
import time
import pickle
import argparse
import warnings
import numpy as np
from sklearn import metrics
import processing_tools as pt
from collections import OrderedDict
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True, linewidth=120, precision=2)

np.random.seed(42)  # 3016222231 76(54)

current_seed = np.random.get_state()[1][0]
with open('granary.txt', 'a+') as o:
    o.write(str(current_seed)+'\n')


def args_parser(args):
    name = args.Name
    run = args.Run
    rho_f = args.Rho_f
    rho_g = args.Rho_g
    num_pivots = args.NumPivots
    range_pivots = args.RangePivots
    with_noise = args.With_noise
    pp = args.PreProcessing
    setting = args.Setting

    return name, run, rho_f, rho_g, num_pivots, range_pivots, with_noise, pp, setting


def stage1(input_stage1, k):

    pivot = input_stage1['pivot']
    clustered_indices = input_stage1['clustered_indices']  # list; i.e. list of cluster's membership vectors

    output_stage1 = OrderedDict()
    output_stage1['clustered_indices'] = clustered_indices
    output_stage1['pivot'] = pivot
    output_stage1['k'] = k
    return output_stage1


def stage2(Y, P, output_stage1, k, rho_f, rho_g):

    N, V = Y.shape
    total_indices = list(range(N))
    output_stage2 = OrderedDict()

    pivot = output_stage1['pivot']
    clustered_indices = output_stage1['clustered_indices']  # all clustered indices except the pivot
    remained_indices = list(set(total_indices).difference(clustered_indices))
    remained_indices.remove(pivot)  # because pivot from stage1 has not been added to clustered indices yet!

    R = len(remained_indices)  # remained entries,

    # print("stage2", k, R)

    ss_p = np.zeros([R, 1])  # Summary Similarity of P
    cv_y = np.zeros([R, 1])  # Summary Similarity of S
    cntr = np.zeros([R, 2])  # cluster Contribution
    index = 0

    for i in remained_indices:
        cv_y[index] = np.sum(np.power(np.mean(Y[[i, pivot], :], axis=0), 2))
        ss_p[index] = np.sum(P[i, pivot] + P[pivot, i] + P[i, i] + P[pivot, pivot])
        Lambda = np.divide(ss_p[index], 2**2)
        cntr[index, :] = (rho_f*cv_y[index]*2 + rho_g*ss_p[index]*Lambda, i)
        index += 1

    argmax = int(np.argmax(cntr[:, 0]))  # k-th cluster's contribution

    clustered_indices_k = list([pivot, int(cntr[argmax, -1])])  # list, membership vector of this cluster
    output_stage2['clustered_indices-' + str(k)] = clustered_indices_k
    output_stage2['clustered_indices'] = clustered_indices  # list of all clustered indices

    return output_stage2


def stage3(Y, P, output_stage2, k, rho_f, rho_g):

    # print("stage 3", k, )
    N, V = Y.shape
    total_indices = list(range(N))
    num_elements = 2

    clustered_indices = output_stage2['clustered_indices']  # all clustered indices
    clustered_indices_k = output_stage2['clustered_indices-' + str(k)]  # pivot + argmax stage2

    cv_y = np.sum(np.power(np.mean(Y[clustered_indices_k, :], axis=0), 2))
    ss_p = np.sum([P[ii, jj] for ii in clustered_indices_k for jj in clustered_indices_k])
    Lambda = np.divide(ss_p, len(clustered_indices_k)**2)
    cntr_old = np.add(rho_f*cv_y*len(clustered_indices_k), rho_g*ss_p*Lambda)

    remained_indices = list(set(total_indices).difference(clustered_indices))

    for i in clustered_indices_k:
        remained_indices.remove(i)  # removing the pivot and the argamx of stage2 from the remaining list

    f_2 = True

    output_stage3 = OrderedDict()

    while f_2 and len(remained_indices) != 0:

        R = len(remained_indices)
        list_of_candidates = [clustered_indices_k + [i] for i in remained_indices if i not in clustered_indices_k]
        cntr = [rho_f * (np.multiply(np.sum(np.power(np.mean(Y[indices, :], axis=0), 2)), len(indices))) +
                rho_g * (np.sum([P[ii, jj] for ii in indices for jj in indices]) *
                         np.divide(np.sum([P[ii, jj] for ii in indices for jj in indices]), len(indices) ** 2))
                for indices in list_of_candidates
                ]

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
            cntr_org = rho_f * (np.multiply(np.sum(np.power(np.mean(Y[clustered_indices_k, :], axis=0), 2)),
                                            len(clustered_indices_k))) + \
                       rho_g * (np.sum([P[ii, jj] for ii in clustered_indices_k for jj in clustered_indices_k])
                                * np.divide(np.sum([P[ii, jj] for ii in clustered_indices_k for jj in
                                                     clustered_indices_k]), len(clustered_indices_k) ** 2))

            tmp_pivot = clustered_indices_k.pop(0)  # removing pivot

            # without pivot
            cntr_norg = rho_f * (np.multiply(np.sum(np.power(np.mean(Y[clustered_indices_k, :], axis=0), 2)),
                                             len(clustered_indices_k))) + \
                        rho_g * (np.sum([P[ii, jj] for ii in clustered_indices_k for jj in clustered_indices_k])
                                 * np.divide(np.sum([P[ii, jj] for ii in clustered_indices_k for jj in
                                                     clustered_indices_k]), len(clustered_indices_k) ** 2))

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
                # argmin = np.argmin(cntr, axis=0)
                # list_of_indices_for_pivot = list(set(list_of_candidates[argmin]).difference(clustered_indices_k))
                # pivot = list_of_indices_for_pivot[-1]
                output_stage3['pivot'] = np.random.choice(remained_indices)

        else:
            output_stage3['continue'] = False
            output_stage3['pivot'] = []
            clustered_indices_k += [j for j in remained_indices if j not in clustered_indices_k]
            clustered_indices += [jj for jj in remained_indices if jj not in clustered_indices]
            output_stage3['clustered_indices-' + str(k)] = clustered_indices_k
            output_stage3['clustered_indices'] = clustered_indices

    return output_stage3


def run_ANomalous_Cluster(pivot, Y, P, rho_f, rho_g):

    # Initialization
    ry, _ = Y.shape
    total_indices = set(range(ry))
    f_1 = True
    k = 0  # Number of Cluster

    input_stage1 = OrderedDict()
    input_stage1['pivot'] = pivot
    input_stage1['clustered_indices'] = []

    clustering_results = OrderedDict()  # to store the total clustering results where key is the k-th cluster__ str(k).

    counter = 0

    while f_1:

        output_stage1 = stage1(input_stage1=input_stage1, k=k,)
        output_stage2 = stage2(Y=Y, P=P, output_stage1=output_stage1, k=k, rho_f=rho_f, rho_g=rho_g)
        output_stage3 = stage3(Y=Y, P=P, output_stage2=output_stage2, k=k, rho_f=rho_f, rho_g=rho_g)
        # print("outstage 3:", output_stage3)

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

            if counter >= 10:  # An extra protection in order to avoid stocking in local optima ry/10

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

    parser.add_argument('--Rho_f', type=float, default=1,
                        help='Feature coefficient during the clustering')

    parser.add_argument('--Rho_g', type=float, default=1,
                        help='Networks coefficient during the clustering')

    parser.add_argument('--NumPivots', type=int, default=1,
                        help='Number of pivots for initialization of clustering process')

    parser.add_argument('--RangePivots', type=int, default=200,
                        help='A Range to chose pivots for initialization of clustering process')

    parser.add_argument('--With_noise', type=int, default=0,
                        help='With noisy features or without')

    parser.add_argument('--PreProcessing', type=str, default='z-m',
                        help='string determining which pre processing method should be applied.'
                             'The first letter determines Y pre processing and the third determines P pre processing. '
                             'Seperated with "-".')

    parser.add_argument('--Setting', type=str, default='all')

    args = parser.parse_args()
    name, run, rho_f, rho_g, num_pivots, range_pivots, with_noise, pp, setting_ = args_parser(args)

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
        print("run:", run, rho_f, rho_g, range_pivots, num_pivots, name, pp, with_noise, setting_)


        def apply_anc(data_type, with_noise):

            # Global initialization
            out_ms = {}
            if setting_ != 'all':
                for setting, repeats in DATA.items():

                    if str(setting) == setting_:

                        print("setting:", setting,)

                        out_ms[setting] = {}
                        for repeat, matrices in repeats.items():
                            print("repeat:", repeat)
                            GT = matrices['GT']
                            Y = matrices['Y']
                            P = matrices['P']
                            Yn = matrices['Yn']
                            N, V = Y.shape

                            # Quantitative case
                            if name.split('(')[0][-1] == 'Q' or name.split('(')[-1] == 'r':
                                _, _, Yz, _, Yrng, _, = pt.preprocess_y(y_in=Y, data_type='Q')

                                if with_noise == 1:
                                    Yn, _, Ynz, _, Ynrng, _, = pt.preprocess_y(y_in=Yn, data_type='Q')

                            # Because there is no Yn in the case of categorical features.
                            if name.split('(')[0][-1] == 'C':
                                enc = OneHotEncoder(sparse=False,)  # categories='auto')
                                Y_oneHot = enc.fit_transform(Y) # oneHot encoding

                                # for WITHOUT follow-up rescale Y_oneHot and for
                                # "WITH follow-up" Y_oneHot should be replaced with Y
                                Y, _, Yz, _, Yrng, _, = pt.preprocess_y(y_in=Y_oneHot, data_type='C')  # Y_oneHot

                            if name.split('(')[0][-1] == 'M':
                                Vq = int(np.ceil(V / 2))  # number of quantitative features -- Y[:, :Vq]
                                Vc = int(np.floor(V / 2))  # number of categorical features  -- Y[:, Vq:]
                                _, _, Yz_q, _, Yrng_q, _, = pt.preprocess_y(y_in=Y[:, :Vq], data_type='Q')
                                enc = OneHotEncoder(sparse=False,)  # categories='auto', )
                                Y_oneHot = enc.fit_transform(Y[:, Vq:])  # oneHot encoding
                                
                                # for WITHOUT follow-up rescale Y_oneHot and for
                                # "WITH follow-up" Y_oneHot should be replaced with Y
                                _, _, Yz_c, _, Yrng_c, _, = pt.preprocess_y(y_in=Y[:, Vq:], data_type='C')  # Y_oneHot
                                Y = np.concatenate([Y[:, :Vq], Y_oneHot], axis=1)
                                Yrng = np.concatenate([Yrng_q, Yrng_c], axis=1)
                                Yz = np.concatenate([Yz_q, Yz_c], axis=1)

                                if with_noise == 1:
                                    Vq = int(np.ceil(V / 2))  # number of quantitative features -- Y[:, :Vq]
                                    Vc = int(np.floor(V / 2))  # number of categorical features  -- Y[:, Vq:]
                                    Vqn = (Vq+Vc)  # the column index of which noise model1 starts

                                    _, _, Ynz_q, _, Ynrng_q, _, = pt.preprocess_y(y_in=Yn[:, :Vq], data_type='Q')
                                    
                                    enc = OneHotEncoder(sparse=False,)  # categories='auto',)
                                    Yn_oneHot = enc.fit_transform(Yn[:, Vq:Vqn])  # oneHot encoding

                                    # for WITHOUT follow-up rescale Yn_oneHot and for
                                    # "WITH follow-up" Yn_oneHot should be replaced with Y
                                    Yn_c, _, Ynz_c, _, Ynrng_c, _, = pt.preprocess_y(y_in=Yn[:, Vq:Vqn], data_type='C')  # Yn_oneHot

                                    Y_ = np.concatenate([Yn[:, :Vq], Yn_oneHot], axis=1)
                                    Yrng = np.concatenate([Ynrng_q, Ynrng_c], axis=1)
                                    Yz = np.concatenate([Ynz_q, Ynz_c], axis=1)

                                    _, _, Ynz_, _, Ynrng_, _, = pt.preprocess_y(y_in=Yn[:, Vqn:], data_type='Q')
                                    Yn_ = np.concatenate([Y_, Yn[:, Vqn:]], axis=1)
                                    Ynrng = np.concatenate([Yrng, Ynrng_], axis=1)
                                    Ynz = np.concatenate([Yz, Ynz_], axis=1)

                            P, _, _, Pu, _, _, Pm, _, _, Pl, _, _ = pt.preprocess_p(p=P)

                            # Pre-processing - Without Noise
                            if data_type == "NP".lower() and with_noise == 0:
                                print("NP")
                                tmp_ms = run_ANomalous_Cluster(pivot=pivot, Y=Y, P=P, rho_f=rho_f, rho_g=rho_g)

                            elif data_type == "z-u".lower() and with_noise == 0:
                                print("z-u")
                                tmp_ms = run_ANomalous_Cluster(pivot=pivot, Y=Yz, P=Pu, rho_f=rho_f, rho_g=rho_g)

                            elif data_type == "z-m".lower() and with_noise == 0:
                                tmp_ms = run_ANomalous_Cluster(pivot=pivot, Y=Yz, P=Pm, rho_f=rho_f, rho_g=rho_g)

                            elif data_type == "z-l".lower() and with_noise == 0:
                                tmp_ms = run_ANomalous_Cluster(pivot=pivot, Y=Yz, P=Pl, rho_f=rho_f, rho_g=rho_g)

                            elif data_type == "rng-u".lower() and with_noise == 0:
                                tmp_ms = run_ANomalous_Cluster(pivot=pivot, Y=Yrng, P=Pu, rho_f=rho_f, rho_g=rho_g)

                            elif data_type == "rng-m".lower() and with_noise == 0:
                                tmp_ms = run_ANomalous_Cluster(pivot=pivot, Y=Yrng, P=Pm, rho_f=rho_f, rho_g=rho_g)

                            elif data_type == "rng-l".lower() and with_noise == 0:
                                tmp_ms = run_ANomalous_Cluster(pivot=pivot, Y=Yrng, P=Pl, rho_f=rho_f, rho_g=rho_g)

                            # Pre-processing - With Noise
                            if data_type == "NP".lower() and with_noise == 1:
                                tmp_ms = run_ANomalous_Cluster(pivot=pivot, Y=Yn, P=P, rho_f=rho_f, rho_g=rho_g)

                            elif data_type == "z-u".lower() and with_noise == 1:
                                tmp_ms = run_ANomalous_Cluster(pivot=pivot, Y=Ynz, P=Pu, rho_f=rho_f, rho_g=rho_g)

                            elif data_type == "z-m".lower() and with_noise == 1:
                                tmp_ms = run_ANomalous_Cluster(pivot=pivot, Y=Ynz, P=Pm, rho_f=rho_f, rho_g=rho_g)

                            elif data_type == "z-l".lower() and with_noise == 1:
                                tmp_ms = run_ANomalous_Cluster(pivot=pivot, Y=Ynz, P=Pl, rho_f=rho_f, rho_g=rho_g)

                            elif data_type == "rng-u".lower() and with_noise == 1:
                                tmp_ms = run_ANomalous_Cluster(pivot=pivot, Y=Ynrng, P=Pu, rho_f=rho_f, rho_g=rho_g)

                            elif data_type == "rng-m".lower() and with_noise == 1:
                                tmp_ms = run_ANomalous_Cluster(pivot=pivot, Y=Ynrng, P=Pm, rho_f=rho_f, rho_g=rho_g)

                            elif data_type == "rng-l".lower() and with_noise == 1:
                                tmp_ms = run_ANomalous_Cluster(pivot=pivot, Y=Ynrng, P=Pl, rho_f=rho_f, rho_g=rho_g)

                            out_ms[setting][repeat] = tmp_ms

                    print("Algorithm is applied on the entire data set!")

            if setting_ == 'all':

                for setting, repeats in DATA.items():

                    print("setting:", setting,)

                    out_ms[setting] = {}
                    for repeat, matrices in repeats.items():
                        print("repeat:", repeat)
                        GT = matrices['GT']
                        Y = matrices['Y']
                        P = matrices['P']
                        Yn = matrices['Yn']

                        N, V = Y.shape

                        # Quantitative case
                        if name.split('(')[0][-1] == 'Q' or name.split('(')[-1] == 'r':
                            _, _, Yz, _, Yrng, _, = pt.preprocess_y(y_in=Y, data_type='Q')

                            if with_noise == 1:
                                Yn, _, Ynz, _, Ynrng, _, = pt.preprocess_y(y_in=Yn, data_type='Q')

                        # Because there is no Yn in the case of categorical features.
                        if name.split('(')[0][-1] == 'C':
                            enc = OneHotEncoder()  # categories='auto')
                            Y = enc.fit_transform(Y)  # oneHot encoding
                            Y = Y.toarray()
                            # Boris's Theory
                            Y, _, Yz, _, Yrng, _, = pt.preprocess_y(y_in=Y, data_type='C')

                        if name.split('(')[0][-1] == 'M':
                            Vq = int(np.ceil(V / 2))  # number of quantitative features -- Y[:, :Vq]
                            Vc = int(np.floor(V / 2))  # number of categorical features  -- Y[:, Vq:]
                            Y_, _, Yz_, _, Yrng_, _, = pt.preprocess_y(y_in=Y[:, :Vq], data_type='M')
                            enc = OneHotEncoder(sparse=False,)  # categories='auto', )
                            Y_oneHot = enc.fit_transform(Y[:, Vq:])  # oneHot encoding
                            Y = np.concatenate([Y_oneHot, Y[:, :Vq]], axis=1)
                            Yrng = np.concatenate([Y_oneHot, Yrng_], axis=1)
                            Yz = np.concatenate([Y_oneHot, Yz_], axis=1)

                            if with_noise == 1:

                                Vq = int(np.ceil(V / 2))  # number of quantitative features -- Y[:, :Vq]
                                Vc = int(np.floor(V / 2))  # number of categorical features  -- Y[:, Vq:]
                                Vqn = (Vq+Vc)  # the column index of which noise model1 starts

                                _, _, Yz_, _, Yrng_, _, = pt.preprocess_y(y_in=Yn[:, :Vq], data_type='M')
                                enc = OneHotEncoder(sparse=False,)  # categories='auto',)
                                Yn_oneHot = enc.fit_transform(Yn[:, Vq:Vqn])  # oneHot encoding
                                Y_ = np.concatenate([Yn_oneHot, Yn[:, :Vq]], axis=1)
                                Yrng = np.concatenate([Yn_oneHot, Yrng_], axis=1)
                                Yz = np.concatenate([Yn_oneHot, Yz_], axis=1)

                                _, _, Ynz_, _, Ynrng_, _, = pt.preprocess_y(y_in=Yn[:, Vqn:], data_type='M')
                                Yn_ = np.concatenate([Y_, Yn[:, Vqn:]], axis=1)
                                Ynrng = np.concatenate([Yrng, Ynrng_], axis=1)
                                Ynz = np.concatenate([Yz, Ynz_], axis=1)

                        P, _, _, Pu, _, _, Pm, _, _, Pl, _, _ = pt.preprocess_p(p=P)

                        # Pre-processing - Without Noise
                        if data_type == "NP".lower() and with_noise == 0:
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, Y=Y, P=P, rho_f=rho_f, rho_g=rho_g)

                        elif data_type == "z-NP".lower() and with_noise == 0:
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, Y=Yz, P=P, rho_f=rho_f, rho_g=rho_g)

                        elif data_type == "z-u".lower() and with_noise == 0:
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, Y=Yz, P=Pu, rho_f=rho_f, rho_g=rho_g)

                        elif data_type == "z-m".lower() and with_noise == 0:
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, Y=Yz, P=Pm, rho_f=rho_f, rho_g=rho_g)

                        elif data_type == "z-l".lower() and with_noise == 0:
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, Y=Yz, P=Pl, rho_f=rho_f, rho_g=rho_g)

                        elif data_type == "rng-u".lower() and with_noise == 0:
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, Y=Yrng, P=Pu, rho_f=rho_f, rho_g=rho_g)

                        elif data_type == "rng-m".lower() and with_noise == 0:
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, Y=Yrng, P=Pm, rho_f=rho_f, rho_g=rho_g)

                        elif data_type == "rng-l".lower() and with_noise == 0:
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, Y=Yrng, P=Pl, rho_f=rho_f, rho_g=rho_g)

                        # Pre-processing - With Noise
                        if data_type == "NP".lower() and with_noise == 1:
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, Y=Yn, P=P, rho_f=rho_f, rho_g=rho_g)

                        elif data_type == "z-u".lower() and with_noise == 1:
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, Y=Ynz, P=Pu, rho_f=rho_f, rho_g=rho_g)

                        elif data_type == "z-m".lower() and with_noise == 1:
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, Y=Ynz, P=Pm, rho_f=rho_f, rho_g=rho_g)

                        elif data_type == "z-l".lower() and with_noise == 1:
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, Y=Ynz, P=Pl, rho_f=rho_f, rho_g=rho_g)

                        elif data_type == "rng-u".lower() and with_noise == 1:
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, Y=Ynrng, P=Pu, rho_f=rho_f, rho_g=rho_g)

                        elif data_type == "rng-m".lower() and with_noise == 1:
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, Y=Ynrng, P=Pm, rho_f=rho_f, rho_g=rho_g)

                        elif data_type == "rng-l".lower() and with_noise == 1:
                            tmp_ms = run_ANomalous_Cluster(pivot=pivot, Y=Ynrng, P=Pl, rho_f=rho_f, rho_g=rho_g)

                        out_ms[setting][repeat] = tmp_ms

                print("Algorithm is applied on the entire data set!")

            return out_ms

        out_ms = apply_anc(data_type=pp.lower(), with_noise=with_noise)

        end = time.time()
        print("Time:", end - start)

        if with_noise == 1:

            name = name + '-N'

        if setting_ != 'all':

            with open(os.path.join('../data', 'FN_' + name + "-" + pp + "-" + setting_ + ".pickle"), 'wb') as fp:
                pickle.dump(out_ms, fp)

        if setting_ == 'all':

            with open(os.path.join('../data', 'FN_' + name + "-" + pp + "-" + ".pickle"), 'wb') as fp:
                pickle.dump(out_ms, fp)

        print("Results are saved!")

        print(" ")
        print("\t", "  p", "  q", " a/e", "\t", "  ARI     ", "  NMI",)
        print(" \t", " \t", " \t", " Ave", " std", " Ave", " std")
        for setting, results in out_ms.items():
            ARI, NMI = [], []
            FNR, TNR = [], []
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

            fnr_ave = np.mean(np.asarray(FNR), axis=0)
            fnr_std = np.std(np.asarray(FNR), axis=0)
            tnr_ave = np.mean(np.asarray(TNR), axis=0)
            tnr_std = np.std(np.asarray(TNR), axis=0)

            print("setting:", setting, "%.3f" % ari_ave, "%.3f" % ari_std, "%.3f" % nmi_ave,
                  "%.3f" % nmi_std)
            print(" ")
            print(" FNR", "TNR")
            print(fnr_ave, fnr_std, tnr_ave, tnr_std)

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
            print("Number of Clusters:", ave_num_clust, std_num_clust)

        # for setting, results in out_ms.items():
        #     ARI, NMI = [], []
        #     for repeat, result in results.items():
        #         tmp_len_ms, tmp_out_ms = sorting_results(result)
        #         tmp_len_gt, tmp_out_gt = sorting_results(DATA[setting][repeat]['GT'])
        #         gt, _ = pt.flat_ground_truth(tmp_out_gt)
        #         lp, _ = pt.flat_clustering_result(tmp_out_ms)
        #         ARI.append(metrics.adjusted_rand_score(gt, lp))
        #         NMI.append(metrics.adjusted_mutual_info_score(gt, lp))
        #
        #     ari_ave_new = np.mean(np.asarray(ARI), axis=0)
        #     ari_std_new = np.std(np.asarray(ARI), axis=0)
        #     nmi_ave_new = np.mean(np.asarray(NMI), axis=0)
        #     nmi_std_new = np.std(np.asarray(NMI), axis=0)
        #     print("setting:", setting, "%.3f" % ari_ave_new, "%.3f" % ari_std_new, "%.3f" % nmi_ave_new,
        #           "%.3f" % nmi_std_new)

    if run == 0:

        print(" \t", " \t", "name:", name)

        with open(os.path.join('../data', name + ".pickle"), 'rb') as fp:
            DATA = pickle.load(fp)

        if with_noise == 1:
            name = name + '-N'
        with open(os.path.join('../data', 'FN_' + name + "-" + pp + "-.pickle"), 'rb') as fp:
            out_ms = pickle.load(fp)

        print(" ")
        print("\t", "  p", "  q", " a/e", "\t", "  ARI     ", "  NMI",)
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
        #         lp, lpi = pt.flat_cluster_results(result)
        #         if not name.split('(')[-1] == 'r':
        #             gt, gti = pt.flat_ground_truth(DATA[setting][repeat]['GT'])
        #         else:
        #             gt = DATA[setting][repeat]['GT']
        #         tmp_cont, _, _ = ev.sk_contingency(tmp_out_ms, tmp_out_gt,)  # N
        #         print("setting:", setting, repeat)
        #         print(tmp_cont)
        #         print(" ")



