import flwr as fl
import tensorflow as tf


## Loading of data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

## Loading the LightGBM Model
model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])


## In line with normal ML algorithm, there will be train and test which is equivalent to fit and evaluate
## Currently is following keras syntax, need to change to lightgbm syntax
## To find out how to manipulate cifarclient to send in the parameters we want from lightgbm model
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}

fl.client.start_numpy_client("[::]:8080", client=CifarClient())