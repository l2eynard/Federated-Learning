import socket
import pickle
import os

from keras.models import load_model
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from keras_pickle_wrapper import KerasPickleWrapper

HOST = 'localhost'  # Standard loopback interface address (localhost)
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)
BUFFER_SIZE = 90000000

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as soc:
    print("Socket is created.")
    soc.bind((HOST, PORT))
    print("Socket bounded to an address & port number. " + str(soc.getsockname()))

    while True:
        soc.listen()
        print("Listening for incoming connection ...")

        connection, address = soc.accept()
        print("Connected to a client: {client_info}.".format(client_info=address))
        with connection:
            print('Connected by', address)

            # Load Base Trained Model
            filename = 'Data/Model/Keras_Model.h5'
            # Instantiate model
            # model = KerasRegressor(build_fn=Sequential(), epochs=10, batch_size=10, verbose=1)
            model = load_model(filename)

            # model = pickle.load(open(filename, 'rb'))
            # model = KerasPickleWrapper(model)
            model = pickle.dumps(model)
            connection.sendall(model)
            print("Model sent to client.")

            received_data = b""
            while str(received_data)[-2] != '.':
                data = connection.recv(BUFFER_SIZE)
                received_data += data

            # Receives updated model from client
            model = pickle.loads(received_data)
            print(model)
            print("Received model from the client.")

            # Saves model to storage
            # filename = 'Data/Model/LGBM_Model.sav'
            # pickle.dump(model, open(filename, 'wb'))
            # print("Model updated successfully.")

            try:
                directory = 'Data/Model/'
                if not os.path.exists(directory):
                    os.mkdir(directory)

                # save the model to disk
                filename = 'Keras_Model.h5'
                model.model.save(directory + filename)
                print("Model updated successfully.")

            except:
                print('Model updated unsuccessfully ')
