import socket
import pickle
import lightgbm

HOST = 'localhost'  # Standard loopback interface address (localhost)
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)
BUFFER_SIZE = 9000000

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
            filename = 'Model/LGBM_model.sav'
            model = pickle.load(open(filename, 'rb'))
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
            filename = 'Model/LGBM_Model_Updated.sav'
            pickle.dump(model, open(filename, 'wb'))
            print("Model updated successfully.")