import pandas as pd
import numpy as np
import lightgbm as lgb

import pickle
import gc
import logging
import os


# #  3. Import dataset
#Import dataset
# Change path directory accordingly
train_dataset = pd.read_csv("Data/Federated/Base/dataset_base.csv")
train_dataset = train_dataset.sample(frac=0.1, random_state=1)
test_dataset = pd.read_csv("Data/Federated/Test/dataset_test.csv")
# test_dataset = test_dataset.sample(frac=0.5, random_state=1)

# # 4. Cleaning of data

# ## 4.1 Cleaning Test dataset


#Test
train_dataset['timestamp'] = pd.to_datetime(train_dataset['timestamp']) #Convert timestamp to datatime
train_dataset = train_dataset.sort_values(by=['site_id', 'timestamp']) #short values by site id then timestamp
train_dataset.fillna(method = 'ffill', inplace=True, limit = 24)#forward fill the missing data up to 12 hours
train_dataset.fillna(method = 'bfill', inplace=True, limit = 24)#backfill up to 12 hours

#fill NaN cells, set all NaN floor_count to 1 and year_built using mean but i think its not too important at the moment
train_dataset.fillna({'floor_count':1, 'year_built': int(train_dataset['year_built'].mean())}, inplace=True)

train_dataset = train_dataset.sort_values(by=['site_id', 'timestamp']) #short values by site id then timestamp

train_dataset.fillna(method = 'ffill', inplace=True, limit = 24)#forward fill the missing data up to 12 hours
train_dataset.fillna(method = 'bfill', inplace=True, limit = 24)#backfill up to 12 hours

#Get columns with empty cells, subsequently get mean value based on site_id and populate cell
missing_cols = [col for col in train_dataset.columns if train_dataset[col].isna().any()] 
mean_data_by_site_id = train_dataset.groupby('site_id')[missing_cols].transform('mean')
train_dataset.fillna(mean_data_by_site_id, inplace=True)

#Add hour, time of year, and weekend columns
train_dataset['hour'] = train_dataset['timestamp'].dt.hour
train_dataset['day_of_year'] = (train_dataset['timestamp'] - pd.Timestamp('2016-01-01')).dt.days%365
train_dataset['is_weekend'] = train_dataset['timestamp'].dt.weekday.isin([5,6]).astype(int)

#Test
test_dataset['timestamp'] = pd.to_datetime(test_dataset['timestamp']) #Convert timestamp to datatime
test_dataset = test_dataset.sort_values(by=['site_id', 'timestamp']) #short values by site id then timestamp
test_dataset.fillna(method = 'ffill', inplace=True, limit = 24)#forward fill the missing data up to 12 hours
test_dataset.fillna(method = 'bfill', inplace=True, limit = 24)#backfill up to 12 hours

#fill NaN cells, set all NaN floor_count to 1 and year_built using mean but i think its not too important at the moment
test_dataset.fillna({'floor_count':1, 'year_built': int(test_dataset['year_built'].mean())}, inplace=True)

test_dataset = test_dataset.sort_values(by=['site_id', 'timestamp']) #short values by site id then timestamp

test_dataset.fillna(method = 'ffill', inplace=True, limit = 24)#forward fill the missing data up to 12 hours
test_dataset.fillna(method = 'bfill', inplace=True, limit = 24)#backfill up to 12 hours

#Get columns with empty cells, subsequently get mean value based on site_id and populate cell
missing_cols = [col for col in test_dataset.columns if test_dataset[col].isna().any()] 
mean_data_by_site_id = test_dataset.groupby('site_id')[missing_cols].transform('mean')
test_dataset.fillna(mean_data_by_site_id, inplace=True)

#Add hour, time of year, and weekend columns
test_dataset['hour'] = test_dataset['timestamp'].dt.hour
test_dataset['day_of_year'] = (test_dataset['timestamp'] - pd.Timestamp('2016-01-01')).dt.days%365
test_dataset['is_weekend'] = test_dataset['timestamp'].dt.weekday.isin([5,6]).astype(int)


# # 5. Reduce Memory Usage
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
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
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

test_dataset = reduce_mem_usage(test_dataset)
train_dataset = reduce_mem_usage(train_dataset)


# # 6. Further clean dataset to handle outlier data and to prepare to train model

#We would like to rescale the meter reading column for each building and meter reading to prevent outliers from skewing the results.
#This is a class to achieve that for any chosen groups. It is a modified version of code by Szymon Maszke: 
#https://stackoverflow.com/questions/55601928/apply-multiple-standardscalers-to-individual-groups

from sklearn.base import clone
class GroupTargetTransform:
    def __init__(self, transformation):
        self.transformation = transformation
        self._group_transforms = {} #this library will hold the group transforms

    def _call_with_function(self, X, y, function: str):
        yhat = pd.Series(dtype = 'float32')#this will hold the rescaled target data
        X['target'] = pd.Series(y, index=X.index)
        for gr in X.groupby(self.features):
            n = gr[0] #this is a tuple id for the group
            g_X = gr[1] #this is the group dataframe
            g_yhat = getattr(self._group_transforms[n], function)(g_X['target'].values.reshape(-1,1))#scale the target variable
            g_yhat = pd.Series(g_yhat.flatten(), index = g_X.index)
            yhat = yhat.append(g_yhat)
        X.drop('target', axis=1, inplace = True)
        return yhat.sort_index()
    
    def fit(self, X, y, features):
        self.features = features
        X['target'] = pd.Series(y, index=X.index) 
        for gr in X.groupby(self.features):
            n = gr[0] #this is a tuple id for the group
            g_X = gr[1] #this is the group dataframe
            sc = clone(self.transformation) #create a new instance of the transform
            self._group_transforms[n] = sc.fit(g_X['target'].values.reshape(-1,1))
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
test_dataset['meter_reading'] = scaler.fit_transform(test_dataset, test_dataset['meter_reading'], ['building_id', 'meter'])
# convert to log(y+1) so the RMSE evaluation metric is actually giving the RMSLE
train_dataset['meter_reading'] = np.log1p(train_dataset['meter_reading']) 
test_dataset['meter_reading'] = np.log1p(test_dataset['meter_reading'])

# # 7. Test Model

from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error, mean_squared_error

#Drop timestamp
train_dataset = train_dataset.drop("timestamp", axis = 1)
test_dataset = test_dataset.drop("timestamp", axis = 1)

pickle.dump(test_dataset, open( "Data/test_dataset.p", "wb" ))
# take 10% to speed up prediction
# discard, test_dataset = train_test_split(test_dataset, test_size = 0.1, random_state=13)

#prepare training data
X = train_dataset.copy()
# X = X.dropna(subset=['meter_reading']) #drop all rows where the meter reading is not included
y = train_dataset['meter_reading'].copy()

X_test = test_dataset.copy()
# X_test = X_test.dropna(subset=['meter_reading']) #drop all rows where the meter reading is not included
y_test = test_dataset["meter_reading"].copy()

#Remove meter_reading so that it does not have the "answers"
del X['meter_reading']
del X['site_id']

del X_test['meter_reading']
del X_test['site_id']

# 80% train, 20% eval
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size = 0.2, random_state=13)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbose': 0,
    "max_depth": 8,
    "num_leaves": 192,
    "max_bin": 512,
    "n_estimators": 1000,
    "early_stopping_rounds": 1000
}

model = lgb.LGBMRegressor(**params)

model.fit(X_train, y_train,
        eval_set=[(X_eval, y_eval)],
        eval_metric='l1')

def clip(x):
    return np.clip(x, a_min=0, a_max=None)

y_predicted = clip(model.predict(X_test, num_iteration=model.best_iteration_))
y_expected = y_test

from sklearn import metrics
#  R2 Score (coefficient of determination) regression score function. Best score 1.0 | Worst score 0.0
print('R2 Score: ', round((metrics.r2_score(y_expected, y_predicted)), 5))
# RMLSE
print('The RMLSE of prediction is:', round(np.sqrt(mean_squared_log_error(y_predicted, y_expected)), 5))

directory = 'Data/Model/'
if not os.path.exists(directory):
    os.mkdir(directory)

# save the model to disk
filename = 'Data/Model/LGBM_Model.sav'
pickle.dump(model, open(filename, 'wb'))