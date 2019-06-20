#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

from flask import Flask, render_template, request
#from tftest import get_config
from server.server import Server
#import server.federation

import json


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

import pandas

import numpy as np


app = Flask(__name__)


# load the model, and pass in the custom metric function
graph = tf.get_default_graph()


server = Server()

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get-config")
def return_config():
    return get_config()

@app.route("/create-federation", methods=['GET'])
def create_federation():
    server.create_federation()
    return render_template('index.html')

@app.route("/print-federation", methods=['GET'])
def print_federation():
    #server.print_federation()
    return render_template('index.html', message=str(server.federation))

@app.route("/print-device", methods=['GET'])
def print_device():
    device_info = server.federation.get_device_config()
    return render_template('index.html', message=device_info)
    
@app.route("/train-federation", methods=['POST'])
def train_federation():
    n_rounds = int(request.form['n_rounds'])
    global graph

    with graph.as_default():
        server.train_federation(n_rounds)

    #results = server.get_results()
    results = "Training emmences"
    return render_template('index.html', message=results)

@app.route("/tf-test", methods=['GET'])
def train_tf():
    digits_mnist = keras.datasets.mnist

    (X_train, y_train), (X_test, y_test) = digits_mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
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
    app.run(debug=False, host='0.0.0.0')