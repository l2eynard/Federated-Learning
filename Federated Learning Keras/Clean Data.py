import pandas as pd
import os

# Import dataset
building_metadata = pd.read_csv("Data/Raw/building_metadata.csv")
primary_use_metadata = pd.read_csv("Data/Processed/primary_use_metadata.csv")

train = pd.read_csv("Data/Raw/train.csv")
weather_train = pd.read_csv("Data/Raw/weather_train.csv")
test = pd.read_csv("Data/Raw/test.csv")
weather_test = pd.read_csv("Data/Raw/weather_test.csv")

# Dupe dataset
building_metadata_C = building_metadata.copy()
train_C = train.copy()
weather_train_C = weather_train.copy()
primary_use_metadata_C = primary_use_metadata.copy()
test_C = test.copy()
weather_test_C = weather_test.copy()

# Remove primary_use with the id
building_metadata_C = pd.merge(building_metadata_C, primary_use_metadata_C, on=['primary_use'])
building_metadata_C = building_metadata_C.drop(columns=['primary_use'])

building_metadata_C.fillna({'floor_count': 1, 'year_built': int(building_metadata['year_built'].mean())}, inplace=True)

# Combine CSVs
combined_dataset_train = train_C.merge(building_metadata_C, how='left', on=['building_id'], validate='many_to_one') \
    .merge(weather_train_C, how='left', on=['site_id', 'timestamp'], validate='many_to_one')


def createdirectory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


directory = 'Data/Processed/'
createdirectory(directory)
subdir = os.path.join(directory, 'dataset_train' + '.csv')
combined_dataset_train.to_csv(subdir, index=False)
