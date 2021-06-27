import flwr as fl
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np

import lightgbm as lgb
from lightgbm import LGBMRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

# Load and compile keras model
# model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
# model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

model = joblib.load("/Users/rey/Downloads/LGBM_Model.sav")

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'learning_rate': 0.005,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbose': 0,
    "max_depth": 8,
    "num_leaves": 128,
    "max_bin": 512,
    "n_estimators": 1000
}
# Load dataset
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Change path directory accordingly
train_dataset = pd.read_csv("/Users/rey/Downloads/Site_10.csv")

# Clean dataset
# TBC

# to make any negative values from prediction to 0
def clip(x):
    return np.clip(x, a_min=0, a_max=None)

# We would like to rescale the meter reading column for each building and meter reading to prevent outliers from skewing the results.
# This is a class to achieve that for any chosen groups. It is a modified version of code by Szymon Maszke:
# https://stackoverflow.com/questions/55601928/apply-multiple-standardscalers-to-individual-groups
from sklearn.base import clone
class GroupTargetTransform:
    def __init__(self, transformation):
        self.transformation = transformation
        self._group_transforms = {}  # this library will hold the group transforms

    def _call_with_function(self, X, y, function: str):
        yhat = pd.Series(dtype='float32')  # this will hold the rescaled target data
        X['target'] = pd.Series(y, index=X.index)
        for gr in X.groupby(self.features):
            n = gr[0]  # this is a tuple id for the group
            g_X = gr[1]  # this is the group dataframe
            g_yhat = getattr(self._group_transforms[n], function)(
                g_X['target'].values.reshape(-1, 1))  # scale the target variable
            g_yhat = pd.Series(g_yhat.flatten(), index=g_X.index)
            yhat = yhat.append(g_yhat)
        X.drop('target', axis=1, inplace=True)
        return yhat.sort_index()

    def fit(self, X, y, features):
        self.features = features
        X['target'] = pd.Series(y, index=X.index)
        for gr in X.groupby(self.features):
            n = gr[0]  # this is a tuple id for the group
            g_X = gr[1]  # this is the group dataframe
            sc = clone(self.transformation)  # create a new instance of the transform
            self._group_transforms[n] = sc.fit(g_X['target'].values.reshape(-1, 1))
        X.drop('target', axis=1, inplace=True)
        return self

    def transform(self, X, y):
        return self._call_with_function(X, y, "transform")

    def fit_transform(self, X, y, features):
        self.fit(X, y, features)
        return self.transform(X, y)

    def inverse_transform(self, X, y):
        return self._call_with_function(X, y, "inverse_transform")

#rescale the target variable for each building and meter type.
from sklearn.preprocessing import MinMaxScaler

scaler = GroupTargetTransform(MinMaxScaler(feature_range = (0,2000))) #2000 is roughly the average meter reading for all the train data
train_dataset['meter_reading'] = scaler.fit_transform(train_dataset, train_dataset['meter_reading'], ['building_id', 'meter'])
# convert to log(y+1) so the RMSE evaluation metric is actually giving the RMSLE
train_dataset['meter_reading'] = np.log1p(train_dataset['meter_reading'])

X = train_dataset
X = train_dataset.dropna(subset=['meter_reading']) #drop all rows where the meter reading is not included
y = train_dataset["meter_reading"]
#Remove meter_reading so that it does not have the "answers"
del X['meter_reading']

# remove meter_reading from dataset
del train_dataset["meter_reading"]

# 80% train, 20% eval
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size = 0.2, random_state=13)

# using the 80% train, I take out 10% for evaluation of accuracy
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state=12)

### Define Client
class CifarClient(fl.client.NumPyClient):

    # def get_parameters(self):
    #     return model.get_weights()

    # def fit(self, parameters, config):
    #     model.set_weights(parameters)
    #     model.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)
    #     return model.get_weights(), len(x_train), {}

    # X_train, y_train | Data, Value
    # X_eval, y_eval | Data, Value
    # X_test, y_test | Data, Value

    #
    def fit(self):
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_eval, y_eval)],
                  eval_metric='l1',
                  early_stopping_rounds=1000)
        return model

    # def evaluate(self, parameters, config):
    #     model.set_weights(parameters)
    #     loss, accuracy = model.evaluate(x_test, y_test)
    #     return loss, len(x_test), {"accuracy": accuracy}

    def evaluate(self):
        model = self.fit()
        y_pred = clip(model.predict(X_test, num_iteration=model.best_iteration_))
        # RMSE
        RMSE = round(mean_squared_log_error(y_pred, y_test) ** 0.5, 5)
        # Return whatever you want
        return model, RMSE, y_pred

# Start client
fl.client.start_numpy_client("[::]:8080", client=CifarClient())