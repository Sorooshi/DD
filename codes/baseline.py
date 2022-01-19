import numpy as np

regression = False


def instantiate_fit_baseline_model(y_train, y_test, target_is_org):

    # model and history are added for the compatibility issue
    history = None
    model = None
    mins = np.min(y_train, axis=0)
    maxs = np.max(y_train, axis=0)
    if target_is_org == 1. and regression is True:
       t = np.arange(mins, maxs, 1)
       y_pred = np.random.choice(t, y_test.shape[0])
    elif target_is_org != 1. and regression is True:
       t = np.linspace(mins, maxs, )
       y_pred = np.random.choice(t, y_test.shape[0])
    else:
        y_pred = np.random.binomial(1, 0.5, size=y_test.shape[0])

    return model, history, y_pred


