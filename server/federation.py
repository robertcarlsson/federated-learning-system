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



class Federation:
    def __init__(self, id=0):
        self.id = id
        self.model = None
        self.devices = []

        self.instantiate_model()
        self.setup_federated_learning()

    def __str__(self):
        return "Federation id:" + str(self.id) + "model:" + self.model.to_json()

    def instantiate_model(self):
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ]) 

    def get_model_config(self):
        return self.model.to_json()

    def get_device_config(self):
        return self.devices[0].get_model_config()

    def setup_federated_learning(self):
        self.devices.append(Device(keras_config=self.model.to_json()))

class Device:
    def __init__(self, keras_config):
        self.model = model_from_json(keras_config)

    def get_model_config(self):
        return self.model.to_json()


