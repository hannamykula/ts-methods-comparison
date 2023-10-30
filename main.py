import time

from sklearn.preprocessing import StandardScaler

from methods.multirocket.multirocket import MultiRocket
from methods.rocket.rocket_functions import generate_kernels, apply_kernels, apply_kernels_v2
from methods.minirocket.minirocket import fit, transform
from methods.inceptiontime.classifiers.inception import Classifier_INCEPTION
# from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np

AVAILABLE_METHODS = ["rocket-ppv-max", "rocket-avg", "minirocket", "multirocket", "inceptiontime"]
method = "minirocket"
kernels_test = [100, 1000, 5000, 10000]
# kernels_test = [1000]
WINDOW = 30
LAG = 20
# set which ts-features to include in rocket
# 1 - avg
# 2 - avg+std
# 4 - avg+std, min, max
NUM_FEATURES = 4



def lag_ts(x, lag, z=1):
    assert len(x.shape) == 1

    X = []
    y = []

    for i in range(0, len(x), z):
        if (i + lag) >= len(x):
            break
        X.append(x[i:(i + lag)].reshape(1, -1))
        y.append(x[(i + lag)])

    X = np.concatenate(X, axis=0)
    y = np.array(y)

    return X, y


if __name__ == "__main__":
    result = ""
    for NUM_KERNELS in kernels_test:
        train_data = pd.read_csv("./data/Processed_DJI train.csv")
        test_data = pd.read_csv("./data/Processed_DJI test.csv")
        train_x = train_data['Close'].to_numpy()
        test_x = test_data["Close"].to_numpy()

        X_train, y_train = lag_ts(train_x, lag=LAG)
        X_test, y_test = lag_ts(test_x, lag=LAG)

        X_training_transform = None
        X_test_transform = None
        error = None
        if method == "rocket-ppv-max":
            # generate random kernels
            kernels = generate_kernels(X_train.shape[-1], NUM_KERNELS)
            # transform training set
            X_training_transform = apply_kernels(X_train, kernels)
            # transform test set
            X_test_transform = apply_kernels(X_test, kernels)
        elif method == "rocket-avg":
            # generate random kernels
            kernels = generate_kernels(X_train.shape[-1], NUM_KERNELS)
            # transform training set
            X_training_transform = apply_kernels_v2(X_train, NUM_FEATURES, kernels)
            # transform test set
            X_test_transform = apply_kernels_v2(X_test, NUM_FEATURES, kernels)
        elif method == "minirocket":
            X_train = np.float32(X_train)
            X_test = np.float32(X_test)
            parameters = fit(X_train)
            X_training_transform = transform(X_train, parameters)
            X_test_transform = transform(X_test, parameters)
        elif method == "multirocket":
            regression = MultiRocket(
                verbose=True
            )
            print("Created multirocket.")
            yhat_train = regression.fit(
                X_train, y_train,
                predict_on_train=False
            )
            predictions = regression.predict(X_test)
            error = mean_absolute_error(y_test, predictions)
        elif method == "inceptiontime":
            scaler = StandardScaler()

            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            input_shape = X_train.shape[1:]

            inceptiontime = Classifier_INCEPTION(
                "./outputs",
                input_shape,
                nb_classes=1,
                verbose=True,
                build=True,
                nb_epochs=500,
            )
            inceptiontime.fit(X_train, y_train, X_test, y_test, y_test)
            predictions = inceptiontime.predict(X_test, y_test, return_df_metrics=False)
            error = mean_absolute_error(y_test, predictions)
        else:
            X_training_transform = X_train
            X_test_transform = X_test

        if not method in ["multirocket", "inceptiontime"]:
            regressor = MLPRegressor(max_iter=300)
            regressor.fit(X_training_transform, y_train)

            predictions = regressor.predict(X_test_transform)
            error = mean_absolute_error(y_test, predictions)
        result_string = \
            f"""Test set-up: \n"
        Method - {method} \n
        Lag - {LAG} \n
        Num. kernels - {NUM_KERNELS} \n\n
        MAE: {error} \n\n"""

        result += result_string

        print(result_string)

    with open(f"result_all_features_{int(time.time())}.txt", "w") as file:
        file.write(result)
