# Clean dataset & Train Base Model (LightGBM)

### Pre-requisite
- Python 3.7
- Pycharm, Jupyter Notebook, etc...
- pandas
- numpy
- lightgbm
- pickle

- Please ensure that the following files can be found on the root directory, Data/Raw folder (create if doesn't exist). building_metadata.csv, train.csv, weather_train.csv, primary_use_metadata.csv.

Dataset can be found at: https://www.kaggle.com/c/ashrae-energy-prediction/data


### How to use?
1. Following the labelling of the python file, it should be executed one after the other, in the following order:
	1. Clean Dataset.ipynb
	2. Train Base Model.ipynb

### What will happen?:
1. The Clean Dataset.ipynb will process the raw dataset, and outputed to your local directory.
2. The Train Base Model.ipynb will subsequently pick up the processed dataset and further process it to train model using LightGBM for this case.
3. Once completed, the updated model will outputted to your local directory.
