# import sys
# sys.path.append("../codes")

import flow as nf
import nn_regressions as nnr
import other_regressions as otr


def instantiate_fit_model(learning_method, alg_name, loss, n_units, input_dim,
                          output_dim, batch_size, n_epochs, learning_rate,
                          x_train, y_train, x_val, y_val, n_estimators, optimizer,):

    if learning_method == "regression":

        if alg_name.lower() == "vnn_reg":
            loss_fn = nnr.determine_tf_loss(loss=loss)
            _model = nnr.VNNRegression(n_units=n_units, input_dim=input_dim, output_dim=output_dim)

        elif alg_name.lower() == "dnn_reg":
            loss_fn = nnr.determine_tf_loss(loss=loss)
            _model = nnr.DNNRegression(n_units=n_units, input_dim=input_dim, output_dim=output_dim)

        elif alg_name.lower() == "nf_reg":
            model = nf.NFFitter(var_size=output_dim, cond_size=input_dim, batch_size=batch_size,
                                n_epochs=n_epochs, lr=learning_rate)
            model.fit(x_train, y_train)

            history = None

        elif alg_name.lower() == "rfr" or alg_name.lower() == "gbr" or \
                alg_name.lower() == "ar" or alg_name.lower() == "lr":
            model = otr.apply_a_regressor(alg_name=alg_name,
                                          n_estimators=n_estimators,
                                          x_train=x_train, y_train=y_train)
            history = None

        elif alg_name.lower() == "gpr":
            # ss_idx = np.random.randint(low=0, high=x_train.shape[0], size=20000)  # because of memory issue
            model = otr.apply_a_regressor(alg_name=alg_name,
                                          n_estimators=n_estimators,
                                          x_train=x_train, y_train=y_train)
            history = None

        else:
            _model = None
            history = None
            print("Undefined model.")

        if alg_name.lower() == "vnn_reg" or alg_name.lower() == "dnn_reg":
            model, history = nnr.compile_and_fit(model=_model, optimizer=optimizer,
                                                 loss=loss_fn, learning_rate=learning_rate,
                                                 batch_size=batch_size, n_epochs=n_epochs,
                                                 x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, )

    elif learning_method == "classification":
        print("classification")

    elif learning_method == "clustering":
        print("clustering")

    else:
        f = True
        print("Wrong learning method")
        assert f is True

    return model, history
