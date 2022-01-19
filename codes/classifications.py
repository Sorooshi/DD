import flow as nf
import tf_neural_networks as tf_nn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel


def apply_a_classifier(alg_name, n_estimators, x_train, y_train):

    alg_name = alg_name.lower()

    # Random Forest Regressor
    if alg_name == "rfc":
        print("Random Forest!")

        model = RandomForestClassifier(n_estimators=n_estimators,
                                       criterion='gini',
                                       min_samples_leaf=1,
                                       verbose=1,
                                       n_jobs=-2,
                                       )

    # Gaussian Process Regressor
    elif alg_name == "gpc":
        print("Gaussian Process!")
        kernel = DotProduct() + WhiteKernel()
        model = GaussianProcessClassifier(kernel=kernel,)

    # Gradient Boosting Regressor
    elif alg_name == "gbc":   # does not support 2d y
        print("Gradient Boosting!")
        model = GradientBoostingClassifier(n_estimators=n_estimators,
                                           verbose=1,
                                           loss='deviance',
                                           )

    # Adaboost Regressor
    elif alg_name == "ac":  # does not support 2d y
        print("Adaboost!")
        model = AdaBoostClassifier(n_estimators=n_estimators,)

    else:
        print ("Undefined regression model.")
        f = True
        assert f is True

    model.fit(x_train, y_train)

    return model


def instantiate_fit_cls_model(alg_name, loss, n_units, input_dim,
                              output_dim, batch_size, n_epochs, learning_rate,
                              x_train, y_train, x_val, y_val, n_estimators, optimizer,):

    if alg_name.lower() == "vnn_cls":
        loss_fn = tf_nn.determine_reg_tf_loss(loss=loss)
        _model = tf_nn.VNNClassification(n_units=n_units, input_dim=input_dim, output_dim=output_dim)

    elif alg_name.lower() == "dnn_cls":
        loss_fn = tf_nn.determine_reg_tf_loss(loss=loss)
        _model = tf_nn.DNNClassification(n_units=n_units, input_dim=input_dim, output_dim=output_dim)

    elif alg_name.lower() == "nf_cls":
        model = nf.NFFitter(var_size=output_dim, cond_size=input_dim, batch_size=batch_size,
                            n_epochs=n_epochs, lr=learning_rate)
        model.fit(x_train, y_train)

        history = None

    elif alg_name.lower() == "rfc" or alg_name.lower() == "gbc" or \
            alg_name.lower() == "ac":
        model = apply_a_classifier(alg_name=alg_name,
                                   n_estimators=n_estimators,
                                   x_train=x_train, y_train=y_train,
                                   )
        history = None

    elif alg_name.lower() == "gpc":
        # ss_idx = np.random.randint(low=0, high=x_train.shape[0], size=20000)  # because of memory issue
        model = apply_a_classifier(alg_name=alg_name,
                                   n_estimators=n_estimators,
                                   x_train=x_train, y_train=y_train)
        history = None

    else:
        _model = None
        history = None
        print("Undefined model.")
        f = True
        assert f is True

    if alg_name.lower() == "vnn_cls" or alg_name.lower() == "dnn_cls":
        model, history = tf_nn.compile_and_fit(model=_model, optimizer=optimizer,
                                               loss=loss_fn, learning_rate=learning_rate,
                                               batch_size=batch_size, n_epochs=n_epochs,
                                               x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                                               )

    return model, history
