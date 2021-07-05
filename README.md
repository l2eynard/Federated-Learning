			# -- Table of Content --


###- 'root' folder
	Federated Learning using Flower framework
			
###- Federated Learning Socket folder 
	Federated Learning using socket programming

###- Model folder 
	Clean raw dataset from Kaggle
	Train Base Model (LightGBM)

------------------------------------------------------------------------------------------------------------------------

## Federated Learning using FLower framework

Run on Python 3.6 and above

1. run trainingbase.py to get the model to save for loading

1. Ensure Tensorflow and Flwr extensions are installed

2. Run server file in terminal (python server.py)

3. Open 2 other terminal and run Client file next on both client files (python client.py)

4. FL should start running

Current difficulties faced:
- The Flower framework used for federated learning not very well documented, not easy to understand. 
	- In addition, difficulty placing model such as LightGBM as it uses different algorithms.
	- Lack of resources online for flower, relatively new.
- Developing a better model for use with the FL system

Documentation:
https://flower.dev/docs/example-pytorch-from-centralized-to-federated.html

