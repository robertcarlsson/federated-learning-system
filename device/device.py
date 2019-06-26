
import requests
from time import sleep

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

class Device:
    def __init__(self, name):
        self.name = name
        
    def get_model_config(self):
        return self.model.to_json()
    
    def set_model_config(self, keras_config_json):
        self.model = model_from_json(keras_config_json)
        self.model.compile(optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

    def set_training_data(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def set_global_weights(self, global_weights):
        self.model.set_weights(global_weights)

    def train_model(self):
        self.model.train_on_batch(self.X_train, self.y_train)

    def get_model_weights(self):
        return self.model.get_weights()

    def run(self):
        while True:
            sleep(2)
            r = requests.get('http://192.168.42.141:5001')
            print("Status: ", r.status_code, "\nHeaders: ", r.headers['content-type'])

