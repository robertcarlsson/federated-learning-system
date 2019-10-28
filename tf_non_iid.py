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
    print ("This file needs these arguments:\n")
    print ("n_datapoints \t\tNumber of datapoints \nn_device \t\tNumber of devices \nn_rounds \t\tNumber of rounds \nshared_init \t\tInitialization option, \"no\" or \"yes\" \n")
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

if (sys.argv[4] == 'not_shared' or 'no'):
    shared_init = False

sorted_indicies = train_labels.argsort()

sorted_train_images = train_images[sorted_indicies]
sorted_train_labels = train_labels[sorted_indicies]

print("Test run ", sorted_train_labels[0:10])


# Now we create each devices local data set
train_images_array = np.split(sorted_train_images[:n_datapoints], n_devices)
train_labels_array = np.split(sorted_train_labels[:n_datapoints], n_devices)

print("Old style", len(train_images_array), len(train_images_array[0]), len(train_images_array[0][0]) ,len(train_images_array[0][0][0]))

numbers = np.arange(10)

#print("numbers: ", numbers)

all_zero = np.argwhere(train_labels == 0)


# Something is a miss when creating the images, it gets another layer 
# 2000, 1, 28, 28 => 2000 images, one image, 28x28 pixels.
# I need to remove the 1 layer.
all_numbers_images = []
all_numbers_labels = []

for number in numbers:

    indices = np.argwhere(train_labels == number)
    #print("TEST2", len(train_images[indices][0]))
    all_numbers_images.append(np.squeeze(train_images[indices]))
    all_numbers_labels.append(np.squeeze(train_labels[indices]))
    #print("lengths: ", len(all_numbers_images[number-1]), len(all_numbers_labels[number-1]))

indices = np.argwhere(train_labels == 1)
test = train_images[indices]
#print("Indicies:", indices)
test = np.squeeze(test)
print("TEST TEST", np.squeeze(test).shape)
print("Test style:", len(test), len(test[0]), len(test[0][0]))

print("All style", len(all_numbers_images), len(all_numbers_images[0]), len(all_numbers_images[0][0]), len(all_numbers_images[0][0][0]))

devices_chosen_digits = []

digits = np.arange(10)

#print(digits)
#print(np.delete(digits, 2))

arr = digits

def random_index(length):
    if length == 0: return 0
    return np.random.randint(0, length)

#for _ in range(10):
#    print("random:", random_digit(4))

for device in range(n_devices):
    devices_chosen_digits.append([])

    #print(arr)
    index = random_index(len(arr)-1)
    chosen_digit = arr[index]
    devices_chosen_digits[device].append(chosen_digit)

    #print(arr, chosen_digit, index)
    arr = np.delete(arr, index)

    index = random_index(len(arr)-1)
    chosen_digit = arr[index]
    devices_chosen_digits[device].append(chosen_digit)
    arr = np.delete(arr, index)


#print("devices", devices_chosen_digits)

train_images_array = []
train_labels_array = []

index = 0

amount_of_points = 1000

for device in devices_chosen_digits:
    train_images_array.append([])
    train_labels_array.append([])
    for digit in device:
        #print("DIGIT:",digit)
        train_images_array[index].extend(all_numbers_images[digit][:amount_of_points])
        train_labels_array[index].extend(all_numbers_labels[digit][:amount_of_points])
    index += 1

#print("length of zero list", len(all_zero), all_zero[101])

#print("LENGTH", len(train_images_array[0]))
#train_images_array = np.split(train_images[:n_datapoints], n_devices)
#train_labels_array = np.split(train_labels[:n_datapoints], n_devices)

print("New styles", len(train_images_array),len(train_images_array[0]), len(train_images_array[0][0]), len(train_images_array[0][0][0]))
#print(train_images_array[0])

devices = []

for i in range(n_devices):
    devices.append(Device(model.to_json(), np.array(train_images_array[i]), np.array(train_labels_array[i])))

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
    
plt.plot(results)
plt.axis([0,n_rounds-1, 0.0,1.0,])

plt.xlabel('Number of rounds')
plt.ylabel('Accuracy')

extra = 'Shared initialization'

if not shared_init:
    extra = 'Individual initialization'

plt.title('Federated Average ANN - Non-IID')

labels = []
for n in range(n_devices):
    labels.append('Device' + str(n+1))

labels.append('Global Model')

plt.legend(labels)

plt.show()
