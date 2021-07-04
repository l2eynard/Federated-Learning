import socket
import pickle
import pandas as pd
import numpy as np
import lightgbm

HOST = 'localhost'  # The server's hostname or IP address
PORT = 65432        # The port used by the server
BUFFER_SIZE = 9000000

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as soc:
    print("Socket is created.")
    soc.connect((HOST, PORT))
    print("Socket bounded to an address & port number. " + str(soc.getsockname()))

    received_data = b""
    while str(received_data)[-2] != '.':
        data = soc.recv(BUFFER_SIZE)
        received_data += data

    # Receives model from server
    model = pickle.loads(received_data)
    print(model)
    print("Received model from the server.")
    # Load local dataset
    train_dataset = pd.read_csv("Data/site_1.csv")

    # Train using base model
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

    train_dataset = reduce_mem_usage(train_dataset)

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


    # rescale the target variable for each building and meter type.
    from sklearn.preprocessing import MinMaxScaler

    scaler = GroupTargetTransform(
        MinMaxScaler(feature_range=(0, 2000)))  # 2000 is roughly the average meter reading for all the train data
    train_dataset['meter_reading'] = scaler.fit_transform(train_dataset, train_dataset['meter_reading'],
                                                          ['building_id', 'meter'])
    # convert to log(y+1) so the RMSE evaluation metric is actually giving the RMSLE
    train_dataset['meter_reading'] = np.log1p(train_dataset['meter_reading'])

    # %%time
    from sklearn.model_selection import train_test_split
    from lightgbm import LGBMRegressor
    from sklearn.metrics import mean_squared_log_error

    # Drop timestamp
    train_dataset = train_dataset.drop("timestamp", axis=1)

    # prepare training data
    X = train_dataset

    # prepare training data
    X = train_dataset
    X = train_dataset.dropna(subset=['meter_reading'])  # drop all rows where the meter reading is not included
    y = train_dataset["meter_reading"]

    # Remove meter_reading so that it does not have the "answers"
    del X['meter_reading']
    del X['site_id']

    # 80% train, 20% eval
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=13)
    # using the 80% train, I take out 20% for evaluation of accuracy
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=12)

    model.fit(X_train, y_train,
              eval_set=[(X_eval, y_eval)],
              eval_metric='l1',
              early_stopping_rounds=1000)


    def clip(x):
        return np.clip(x, a_min=0, a_max=None)

    # Prediction
    y_pred = clip(model.predict(X_test, num_iteration=model.best_iteration_))

    # Basic RMLSE
    print('The rmse of prediction is:', round(mean_squared_log_error(y_pred, clip(y_test)) ** 0.5, 5))

    # Sends trained model back to server
    model = pickle.dumps(model)
    soc.sendall(model)
    print("Model sent to server.")

    soc.close()
    print("Socket closed.")