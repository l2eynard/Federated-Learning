import pandas as pd
import numpy as np
import os

try:
    # Import dataset
    train_dataset = pd.read_csv("Data/Processed/dataset_train.csv")
except:
    print("dataset_train.csv not found!")

# Split into 5 pieces
shuffled = train_dataset.sample(frac=1)
dataset = np.array_split(shuffled, 5)


def createdirectory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


# Base dataset
directory = 'Data/Federated/Base/'
createdirectory(directory)
subdir = os.path.join(directory, 'dataset_base' + '.csv')
dataset[3].to_csv(subdir, index=False)

# Test dataset
directory = 'Data/Federated/Test/'
createdirectory(directory)
subdir = os.path.join(directory, 'dataset_test' + '.csv')
dataset[4].to_csv(subdir, index=False)

# Client dataset
directory = 'Data/Federated/Client/'
createdirectory(directory)
# Client1
subdir = os.path.join(directory, 'Client' + '.csv')
dataset[0].to_csv(subdir, index=False)
# Client2
subdir = os.path.join(directory, 'Client2' + '.csv')
dataset[1].to_csv(subdir, index=False)
# Client3
subdir = os.path.join(directory, 'Client3' + '.csv')
dataset[2].to_csv(subdir, index=False)
