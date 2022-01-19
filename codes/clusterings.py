from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering


def instantiate_fit_clu_model(alg_name, n_clusters, x_train, y_train):

    alg_name = alg_name.lower()

    if alg_name == "km":
        print("K-Means!")
        model = MiniBatchKMeans(n_clusters=n_clusters,)

    elif alg_name == "gm":
        print("Gaussian Mixture!")
        model = GaussianMixture(n_components=n_clusters, n_init=10, )

    elif alg_name == "agg":
        print("Agglomerative!")
        model = AgglomerativeClustering(n_clusters=n_clusters)

    elif alg_name == "ms":
        print("Mean-Shift!")
        model = MeanShift(bandwidth=None, )

    # methods lead to multiclass clusters
    elif alg_name == "ap":
        print("Affinity Propagation!")
        model = AffinityPropagation()

    elif alg_name == "dbs":
        model = DBSCAN(eps=0.5, min_samples=5, )
        print("DBSCAN!")

    elif alg_name == "sefnac":
        print("SEFNAC!")

    elif alg_name == "nnc":
        print("Neural Network Clustering!")

    else:
        print ("Undefined clustering model.")
        f = True
        assert f is True

    model.fit(x_train)
    history = None

    return model, history





