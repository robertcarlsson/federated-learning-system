from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

# Helper libraries
import numpy as np

graph = tf.get_default_graph()


class Federation:
    def __init__(self, id=0):
        self.id = id
        self.model = None
        self.devices = []

        #tf.logging.set_verbosity(tf.logging.ERROR)

        self.n_devices = 5
        

        self.load_mnist_dataset()
        self.instantiate_model()
        self.setup_federated_learning()


    def __str__(self):
        return "Federation id: " + str(self.id) + "\nmodel: \n" + self.model.to_json()

    def load_mnist_dataset(self):
        digits_mnist = keras.datasets.mnist

        (self.X_train, self.y_train), (self.X_test, self.y_test) = digits_mnist.load_data()
        self.X_train = self.X_train / 255.0
        self.X_test = self.X_test / 255.0

    def instantiate_model(self):
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        self.model.compile(optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

    def get_model_config(self):
        return self.model.to_json()

    def get_device_config(self):
        return self.devices[0].get_model_config()

    def setup_federated_learning(self, n_devices=5):
        self.n_devices = n_devices
        X_train_array = np.split(self.X_train, n_devices)
        y_train_array = np.split(self.y_train, n_devices)

        for i in range(n_devices):
            self.devices.append(Device(self.model.to_json(), X_train_array[i], y_train_array[i]))

    def train_federation_round(self):
        global_weights = self.model.get_weights()

        for device in self.devices:

            device.set_global_weights(global_weights)
            device.train_model()
            device_weights = device.get_model_weights()

            global_weights = [ w * 0 for w  in global_weights]
            global_weights = [ w1 + w2/self.n_devices for w1, w2 in zip(global_weights, device_weights) ]

        self.model.set_weights(global_weights)

    def train_federation_epoch(self, n_rounds):
        global graph
        with graph.as_default():
            self.model.fit(self.X_train, self.y_train, epochs=1)
        """
        self.test_results = []
        for _ in range(n_rounds):
            self.train_federation_round()
            self.test_results.append(self.model.evaluate(self.X_test, self.y_test))
        """


class Device:
    def __init__(self, keras_config, X_train, y_train):
        self.model = model_from_json(keras_config)
        self.model.compile(optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])
        self.X_train = X_train
        self.y_train = y_train

    def get_model_config(self):
        return self.model.to_json()
    
    def set_global_weights(self, global_weights):
        self.model.set_weights(global_weights)

    def train_model(self):
        self.model.train_on_batch(self.X_train, self.y_train)

    def get_model_weights(self):
        return self.model.get_weights()


