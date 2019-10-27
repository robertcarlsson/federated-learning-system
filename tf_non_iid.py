#!/usr/bin/python3
from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Dropout
#from tensorflow.keras.layers import Embedding
#from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

# test commit
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import sys
print ("This is the name of the script: ", sys.argv[0])
print ("Number of arguments: ", len(sys.argv))
print ("The arguments are: " , str(sys.argv))

if (len(sys.argv) < 5):
    print ("This file needs these arguments:")
    print ("python tf_test.py n_datapoints n_device n_rounds shared_init")
    exit()

digits_mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = digits_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

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

class Device:
    def __init__(this, config, X_train, y_train):
        this.model = keras.models.model_from_json(config)
        this.model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
        this.X_train = X_train
        this.y_train = y_train
    
    def train_model(this):
        this.model.fit(this.X_train, this.y_train, epochs=1, verbose=0)
    
    def train_model_batch(this):
        this.model.train_on_batch(this.X_train, this.y_train)

n_datapoints = int(sys.argv[1])
n_devices = int(sys.argv[2])
n_rounds = int(sys.argv[3])

shared_init = True

if (sys.argv[4] == 'not_shared'):
    shared_init = False

train_images_array = np.split(train_images[:n_datapoints], n_devices)
train_labels_array = np.split(train_labels[:n_datapoints], n_devices)


devices = []

for i in range(n_devices):
    devices.append(Device(model.to_json(), train_images_array[i], train_labels_array[i]))

results = []


# Shared initialization
if shared_init:
    global_weights = model.get_weights()
    for device in devices:
        device.model.set_weights(global_weights)

for _ in range(n_rounds):
    
    global_weights = model.get_weights()
    
    for device in devices:
        #device.model.set_weights(global_weights)
        #device.train_model_batch()
        device.train_model()
    
    global_weights = [ w * 0 for w  in global_weights]
    
    for device in devices:
        device_weights = device.model.get_weights()
        global_weights = [ w1 + w2 for w1, w2 in zip(global_weights, device_weights) ]

    global_weights = [ w/n_devices for w  in global_weights]

    
    model.set_weights(global_weights)
    
    models = [ device.model for device in devices]
    models.append(model)
    
    round_results = [ model.evaluate(test_images, test_labels, verbose=0)[1] for model in models]
    results.append(round_results)

    for device in devices:     
        device.model.set_weights(global_weights)
    
results

plt.plot(results)
plt.axis([0,n_rounds-1, 0.0,1.0,])

plt.xlabel('Number of rounds')
plt.ylabel('Accuracy')

extra = 'Shared initialization'

if not shared_init:
    extra = 'Individual initialization'

plt.title('Federated Average ANN - ' + extra)

labels = []
for n in range(n_devices):
    labels.append('Device' + str(n+1))

labels.append('Global Model')

plt.legend(labels)

plt.show()