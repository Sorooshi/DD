from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel


def apply_a_regressor(alg_name, n_estimators, x_train, y_train):
    
    # Random Forest Regressor
    if alg_name.lower() == "rfr":
        print("Random Forest!")

        model = RandomForestRegressor(n_estimators=n_estimators,
                                      criterion='mse',
                                      min_samples_leaf=1,
                                      verbose=1,
                                      n_jobs=-2,
                                      )

    # Gaussian Process Regressor
    elif alg_name.lower() == "gpr":
        print("Gaussian Process!")
        kernel = DotProduct() + WhiteKernel()
        model = GaussianProcessRegressor(kernel=kernel, random_state=0)

    # Gradient Boosting Regressor
    elif alg_name.lower() == "gbr":   # does not support 2d y
        print("Gradient Boosting!")
        model = GradientBoostingRegressor(n_estimators=n_estimators,
                                          verbose=1,
                                          loss='ls',
                                          )

    # Adaboost Regressor
    elif alg_name.lower() == "ar":  # does not support 2d y
        print("Adaboost!")
        model = AdaBoostRegressor(n_estimators=n_estimators,
                                  loss="linear",
                                  )
    elif alg_name.lower() == "lr":
        print ("Linear Regression!")
        model = LinearRegression()
    else:
        print ("Undefined ensembling model.")
        f = True
        assert f is True

    model.fit(x_train, y_train)

    return model