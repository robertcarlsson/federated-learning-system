# federated-learning-system
Federated learning system using docker and tensorflow.


## Running
If you have docker-compose installed you can start a federated learning process by doing these commands.
Create two terminal windows and have one of them run as the federation and the other for the devices.

### 1. Starting the federated server
cd federated-learning-system/federation

sudo docker-compose up

### 2. Starting the devices
Open the Dockerfile at the device folder in the repository, and edit the ip-adress to the ip-adress the federated server will run at. For example your docker-running device network adress.


at this row "CMD [ "/app/app.py" , "192.168.1.105"]" - Change the ip-adress to your own


cd federated-learning-system/device

sudo docker-compose up

Now the devices should be up and running and both the federated server and devices are stuck in the connected/ready face, so we need to give the federation a starting signal.

### 3.
curl http://\*ipadress\*:5001/start-fed
