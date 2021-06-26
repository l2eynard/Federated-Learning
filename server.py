import flwr as fl


## To check if configurations should be set here
fl.server.start_server(config={"num_rounds": 3})
