import os

import flwr as fl
import tensorflow as tf
import pandas as pd
import numpy as np

# Make TensorFlow log less verbose
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


if __name__ == "__main__":
    # Load dataset
    # Cleaning dataset
    train_dataset = pd.read_csv("https://raw.githubusercontent.com/LowTeckWei/itp-se3/main/Energy%20Prediction/Data/dataset_train.csv?token=ANHIIDFOJOI765YKDI2TIJDA4QLOI")
    # End Load Data

    """ 
    Cleaning data
    """
    # Train
    train_dataset['timestamp'] = pd.to_datetime(train_dataset['timestamp'])  # Convert timestamp to datatime
    train_dataset = train_dataset.sort_values(by=['site_id', 'timestamp'])  # short values by site id then timestamp
    train_dataset.fillna(method='ffill', inplace=True, limit=24)  # forward fill the missing data up to 12 hours
    train_dataset.fillna(method='bfill', inplace=True, limit=24)  # backfill up to 12 hours

    # fill NaN cells, set all NaN floor_count to 1 and year_built using mean but i think its not too important at the moment
    train_dataset.fillna({'floor_count': 1, 'year_built': int(train_dataset['year_built'].mean())}, inplace=True)

    train_dataset = train_dataset.sort_values(by=['site_id', 'timestamp'])  # short values by site id then timestamp

    train_dataset.fillna(method='ffill', inplace=True, limit=24)  # forward fill the missing data up to 12 hours
    train_dataset.fillna(method='bfill', inplace=True, limit=24)  # backfill up to 12 hours

    # Get columns with empty cells, subsequently get mean value based on site_id and populate cell
    missing_cols = [col for col in train_dataset.columns if train_dataset[col].isna().any()]
    mean_data_by_site_id = train_dataset.groupby('site_id')[missing_cols].transform('mean')
    train_dataset.fillna(mean_data_by_site_id, inplace=True)

    # Add hour, time of year, and weekend columns
    train_dataset['hour'] = train_dataset['timestamp'].dt.hour
    train_dataset['day_of_year'] = (train_dataset['timestamp'] - pd.Timestamp('2016-01-01')).dt.days % 365
    train_dataset['is_weekend'] = train_dataset['timestamp'].dt.weekday.isin([5, 6]).astype(int)
    """
    End Cleaning Data
    """

    """
    Reducing memory usage
    """
    def reduce_mem_usage(df, verbose=True):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024 ** 2
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        end_mem = df.memory_usage().sum() / 1024 ** 2
        if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                    start_mem - end_mem) / start_mem))
        return df
    """
    End Reducing Memory
    """

    """
    Further refinement of dataset
    """
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
    """
    End Further refinement of dataset
    """

    """
    Rescale Target Variable
    """
    # rescale the target variable for each building and meter type.
    from sklearn.preprocessing import MinMaxScaler

    scaler = GroupTargetTransform(
        MinMaxScaler(feature_range=(0, 2000)))  # 2000 is roughly the average meter reading for all the train data
    train_dataset['meter_reading'] = scaler.fit_transform(train_dataset, train_dataset['meter_reading'],
                                                          ['building_id', 'meter'])
    # convert to log(y+1) so the RMSE evaluation metric is actually giving the RMSLE
    train_dataset['meter_reading'] = np.log1p(train_dataset['meter_reading'])
    """
    End target variable
    """

    """
    Drop timeframe
    """
    train_dataset = train_dataset.drop("timestamp", axis=1)
    """
    End Drop timeframe
    """

    """
    Prepare training data
    """
    # prepare training data
    # create a dataframe with all training data except the target column
    train_x = train_dataset.drop(columns=['meter_reading'])
    train_y = train_dataset["meter_reading"]
    # Remove meter_reading so that it does not have the "answers"
    # del X['meter_reading']
    """
    End training data
    """

    x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2)

    """
    Loading Keras Model
    """
    from keras.models import Sequential
    from keras.layers import Dense

    # create model
    model = Sequential()

    # get number of columns in training data
    n_cols = train_x.shape[1]

    # add model layers
    model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    # compile model using mse as a measure of model performance
    # model.compile(optimizer='adam', loss='mean_squared_error')
    model.compile(optimizer='adam', loss='mean_squared_error', metrics='accuracy')

    from keras.callbacks import EarlyStopping

    # set early stopping monitor so the model stops training when it won't improve anymore
    early_stopping_monitor = EarlyStopping(patience=3)
    """
    End Keras Model
    """


    # Define Flower client
    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            return model.get_weights()

        def fit(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            model.fit(x_train, y_train, validation_split=0.2, epochs=20, callbacks=[early_stopping_monitor])
            return model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            #loss, accuracy = model.evaluate(x_test, y_test)
            loss, accuracy = model.evaluate(x_test, y_test)
            return loss, len(x_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client(server_address="[::]:8080", client=CifarClient())
