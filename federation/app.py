#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
from flask import Flask, render_template, request, jsonify
import json

import numpy as np

app = Flask(__name__)

from federation import Federation

fed = Federation()

@app.route("/", methods=['GET', 'POST'])
def index():
    message = 'jag har kopplat denna fil till min container'
    if request.method == 'POST':
        message = "\nFederation ID: " + request.form['fed_id'] \
            + "\nDevices started: " + request.form['n_devices']
    return render_template('index.html', message=message)

@app.route("/create-federation", methods=['GET'])
def create_federation():
    fed.instantiate_model()
    message = fed.get_model_config()
    return render_template('index.html', message=message)

@app.route("/connect", methods=['GET'])
def connect_device():
    return jsonify(fed.connect_device())

@app.route("/config", methods=['GET'])
def get_config():
    return fed.get_model_config()

@app.route("/data", methods=['GET'])
def get_data():
    return jsonify(fed.send2)

@app.route("/ready", methods=['GET'])
def device_ready():
    device_id = request.args.get('id')
    devices_ready = True
    print("Device ID: {} is ready".format(device_id))     

    for device in fed.connected_devices:
        if device.id == int(device_id):
            device.ready = True
        if not device.ready:
            devices_ready = False
        print('Device id {} checked {} checked id {}'.format(device.id, device.ready, device_id))

    data = {}
    data['weights_update_ready'] = devices_ready

    if devices_ready and fed.first_round:
        fed.first_round = False
        fed.instantiate_model() 
        fed.set_random_weights()

    if devices_ready:
        data['weights'] = [ w.tolist() for w in fed.global_weights ]

    return jsonify(data)

@app.route("/round", methods=['POST'])
def device_round():
    requested_data = request.get_json()
    device_id = requested_data['id']
    weights = requested_data['weights']
    #for arr in weights:
    #    print('Weights: ', len(arr))
    weights = [np.array(w) for w in weights]
    #print('Weights: ', weights)
    for device in fed.connected_devices:
        if device.id == device_id:
            device.weights = weights
            device.round_ready = True
            device.ready = False

    fed.aggregate_function()
    return jsonify("round started")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)