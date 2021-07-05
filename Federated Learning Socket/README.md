# Federated Learning Socket

### Pre-requisite library
- pandas
- numpy
- lightgbm
- pickle

### How to use?
1. run server.py
2. Once server.py has started, run client.py.
3. Once client.py has started, run client2.py.

### What will happen?:
1. The server will listen for any client, once a connection is established between the server and the client, the server will send the trained model to the client. (only 1 connection can be established at any one point of time)
2. Subsequently, the client will load the model and its own site dataset to start training the model.
3. Once completed, the updated model will be sent back to the server and updated.
