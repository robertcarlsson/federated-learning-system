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
        
        #self.load_mnist_dataset()
        #self.instantiate_model()


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
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        self.model.compile(optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

    def get_model_config(self):
        return self.model.to_json()
