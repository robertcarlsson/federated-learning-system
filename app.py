from flask import Flask, render_template, request
from tftest import get_config
from server.server import Server


import json

app = Flask(__name__)


federated_server = Server()

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get-config")
def return_config():
    return get_config()

@app.route("/create-federation", methods=['GET'])
def create_federation():
    federated_server.create_federation()
    return render_template('index.html')

@app.route("/print-federation", methods=['GET'])
def print_federation():
    federated_server.print_federation()
    return render_template('index.html', message=str(federated_server.federation))

@app.route("/print-device", methods=['GET'])
def print_device():
    device_info = federated_server.federation.get_device_config()
    return render_template('index.html', message=device_info)
    

