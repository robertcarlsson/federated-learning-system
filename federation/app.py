#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals


from flask import Flask, render_template, request, jsonify
import json

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

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
    print("Device ID: {} is ready".format(device_id))

        
    devices_ready = True

    for device in fed.connected_devices:
        if device.id == int(device_id):
            device.ready = True
        if device.ready == False:
            devices_ready = False
        print('Device {} checked {} since {}'.format(device.id, device.ready, device_id))

    data = {}
    data['all_ready'] = devices_ready
    if devices_ready and fed.first_round:
        fed.first_round = False
        fed.instantiate_model()
        data['weights'] = fed.get_global_weights()
    elif devices_ready and not fed.first_round:
        data['weights'] = fed.global_weights

    return jsonify(data)

@app.route("/round", methods=['POST'])
def start_round():
    weights = request.form['weights']
    print('Weights: ', weights)
    #weights = [np.array(w) for w in weights]
    #print('Weights: ', weights)
    for device in fed.connected_devices:
        if device.id == int(request.form['id']):
            device.weights = weights
            device.round_ready = True
            device.ready = False

    return jsonify("round started")

@app.route("/tf-test", methods=['GET'])
def train_tf():
    digits_mnist = keras.datasets.mnist

    (X_train, y_train), (X_test, y_test) = digits_mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=1, verbose=0)
    res = model.evaluate(X_test, y_test)
    return render_template('index.html', message=res)



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)