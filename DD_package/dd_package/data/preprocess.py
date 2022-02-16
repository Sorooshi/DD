import numpy as np
from sklearn.preprocessing import MinMaxScaler, \
    StandardScaler, QuantileTransformer, RobustScaler


def range_standardizer(x):
    """ Returns Range standardized datasets set.
            Input: a numpy array, representing entity-to-feature matrix.
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    x_rngs = np.ptp(x, axis=0)
    x_means = np.mean(x, axis=0)

    x_r = np.divide(np.subtract(x, x_means), x_rngs)  # range standardization

    return np.nan_to_num(x_r)


def range_standardizer_(x_test, x_train):
    """ Returns Range standardized datasets set.
    Input: a numpy array, representing entity-to-feature matrix.
    """

    if not isinstance(x_test, np.ndarray):
        x = np.asarray(x_test)

    x_rngs = np.ptp(x_train, axis=0)
    x_means = np.mean(x_train, axis=0)

    x_r = np.divide(np.subtract(x_test, x_means), x_rngs)  # range standardization

    return np.nan_to_num(x_r)


def zscore_standardizer(x):
    """ Returns Z-scored standardized datasets set.
            Input: a numpy array, representing entity-to-feature matrix.
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    x_stds = np.std(x, axis=0)
    x_means = np.mean(x, axis=0)

    x_z = np.divide(np.subtract(x, x_means), x_stds)  # z-scoring

    return np.nan_to_num(x_z)


def zscore_standardizer_(x_test, x_train):
    """ Returns Z-scored standardized datasets set.
            Input: a numpy array, representing entity-to-feature matrix.
    """

    x_stds = np.std(x_train, axis=0)
    x_means = np.mean(x_train, axis=0)

    x_z = np.divide(np.subtract(x_test, x_means), x_stds)  # z-scoring

    return np.nan_to_num(x_z)


def quantile_standardizer(x, out_dist):

    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    QT = QuantileTransformer(output_distribution=out_dist,)
    x_q = QT.fit_transform(x)

    return x_q, QT


def quantile_standardizer_(QT, x,):

    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    x_q = QT.fit_transform(x)

    return x_q


def minmax_standardizer(x):
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    x_mm = np.divide(np.subtract(x, x.min(axis=0)),
                     (x.max(axis=0) - x.min(axis=0)))
    return np.nan_to_num(x_mm)


def minmax_standardizer_(x_test, x_train):
    x_mm = np.divide(np.subtract(x_test, x_train.min(axis=0)),
                     (x_train.max(axis=0) - x_train.min(axis=0)))
    return np.nan_to_num(x_mm)


def robust_standardizer(x):
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    RS = RobustScaler()
    x_rs = RS.fit_transform(x)
    return x_rs, RS


def robust_standardizer_(RS, x):
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    x_rs = RS.fit_transform(x)
    return x_rs


def preprocess_data(x, pp):

    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    if pp == "rng":
        print("pre-processing:", pp)
        x = range_standardizer(x=x)
        print("Preprocessed data shape:", x.shape, )
    elif pp == "zsc":
        print("pre-processing:", pp)
        x = zscore_standardizer(x=x)
        print("Preprocessed data shape:", x.shape,)
    elif pp == "mm":  # MinMax
        print("pre-processing:", pp)
        x = minmax_standardizer(x=x)
        print("Preprocessed data shape:", x.shape,)
    elif pp == "rs":  # Robust Scaler (subtract median and divide with [q1, q3])
        print("pre-processing:", pp)
        x, rs_x = robust_standardizer(x=x)
        print("Preprocessed data shape:", x.shape,)
    elif pp == "qtn":  # quantile_transformation with Gaussian distribution as output
        x, qt_x = quantile_standardizer(x=x, out_dist="normal")
        print("Preprocessed data shape:", x.shape,)
    elif pp == "qtu":  # quantile_transformation with Uniform distribution as output
        x, qt_x = quantile_standardizer(x=x, out_dist="uniform")
        print("Preprocessed data shape:", x.shape,)
    elif pp is None:
        x_org = x
        print("No pre-processing")
    else:
        print("Undefined pre-processing")

    return x





