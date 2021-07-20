# Data Prediction Setup Guide

##  Table of Content 

### 1. Pre-requisite
### 2. Execution Order
### 3. Execution

------------------------------------------------------------------------------------------------------------------------

### 1. Pre-requisite
- Pycharm
- Python 3.7

------------------------------------------------------------------------------------------------------------------------

### 2. Execution Order
Clean Data.py -> Prep Data.py -> Train Model- LGBM.py -> Server.py -> Client.py -> Client2.py -> Client3.py

------------------------------------------------------------------------------------------------------------------------

### 3. Execution

### Clean Data
1. Run Clean Data.py on Pycharm.
Note: Ensure to download dataset if required from https://www.kaggle.com/c/ashrae-energy-prediction/data and place it in root folder '/Data/Raw'. 

### Prep Data
1. Run Clean Data.py on Pycharm.

### Train Model- LGBM
1. Run Train Model- LGBM.py on Pycharm.

### Federated Learning - Socket
Now that all the data has been cleaned and base model has been created. We can attempt federated learning.

1. run server.py. (Should observe that server has started listening...)
2. Once server.py has started, run client.py.
3. Subsequently you can start client2.py & client3.py as well. They will be processed 1 after another.

