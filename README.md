Run on Python 3.6 and above

1. run trainingbase.py to get the model to save for loading

1. Ensure Tensorflow and Flwr extensions are installed

2. Run server file in terminal (python server.py)

3. Open 2 other terminal and run Client file next on both client files (python client.py)

4. FL should start running

Current difficulties faced:
- How to send updated model into server, this is to allow the other clients to use the updated model
- Developing a better model for use with the FL system

Documentation:
https://flower.dev/docs/example-pytorch-from-centralized-to-federated.html