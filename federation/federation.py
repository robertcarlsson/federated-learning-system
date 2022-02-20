# future statements like these are kept for backwards compability
# which I am not sure make much sense since python 3.5 is needed for tensorflow
#from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np

graph = tf.compat.v1.get_default_graph()

class Device:
    def __init__(self, device_id):
        self.id = device_id
        self.round_ready = False
        self.ready = False
        self.weights = None

        self.X_train = None
        self.y_train = None


class Federation:
    def __init__(self, fed_id=0):
        self.id = fed_id
        self.model = None
        self.global_weights = None

        self.load_mnist_dataset()
        self.instantiate_model()

        self.device_id = 0
        self.connected_devices = []

        self.first_round = True
        self.ready = False

    def __str__(self):
        return "Federation id: " + str(self.id) + "\nmodel: \n" + self.model.to_json()

    def connect_device(self):
        device = Device(self.device_id)
        self.connected_devices.append(device)
        self.device_id += 1
        return (self.device_id - 1)

    def set_fed_ready(self):
        self.ready = True
        # initate the data for the devices
        n_devices = len(self.connected_devices)
        points_per_device = 60000 // n_devices
        points_per_device = 2000
        n_points = n_devices * points_per_device

        self.X_used = self.X_train[:n_points]
        self.y_used = self.y_train[:n_points]

        self.X_list = [ X.tolist() for X in np.split(self.X_used, n_devices) ]
        self.y_list = [ y.tolist() for y in np.split(self.y_used, n_devices) ]

    def load_mnist_dataset(self):
        digits_mnist = keras.datasets.mnist

        (self.X_train, self.y_train), (self.X_test, self.y_test) = digits_mnist.load_data()
        self.X_train = self.X_train / 255.0
        self.X_test = self.X_test / 255.0

        #self.X_send = self.X_train[:5000]
        #self.y_send = self.y_train[:5000]

        #self.send2 = {}
        #self.send2['X_train'] = self.X_send.tolist()
        #self.send2['y_train'] = self.y_send.tolist()
        #self.send2 = "sd"
        #self.send = [ (X.tolist(), y.tolist()) for X, y in zip(self.X_send, self.y_send) ]


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

    def set_random_weights(self):
        self.global_weights = self.model.get_weights()

    def get_device_data(self, device_id):
        device_id = int(device_id)
        data = {}
        data['X_train'] = self.X_list[device_id]
        data['y_train'] = self.y_list[device_id]
        return data

    def aggregate_function(self):
        if all([ device.round_ready for device in self.connected_devices ]):
            global_weights = [ w * 0 for w  in self.global_weights]
            for device in self.connected_devices:
                global_weights = [ w1 + w2 for w1, w2 in zip(global_weights, device.weights) ]
            self.global_weights = [ w/len(self.connected_devices) for w in global_weights]
    
    def test_model(self):
        # Train all the device models and the aggregation
        pass

if __name__ == '__main__':
    fed = Federation()
    print("Type: ", type(fed.X_train), "Shape: ", fed.X_train.shape)

