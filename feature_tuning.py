#imports
import cv2
import numpy as np
import os
import pickle
import random

#Hyperparameter imports
import talos as ta
from talos.model import normalizers, hidden_layers
from talos.model.normalizers import lr_normalizer
from keras.activations import relu, elu, sigmoid
from keras.optimizers import Nadam, RMSprop
from keras.losses import logcosh, binary_crossentropy
from keras import layers
from keras.models import Model

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D,Flatten ,Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

path ='/Users/katiekorn/PycharmProjects/ds/bees2/data/'
seed = 1122021
features = []
labels = []

def read_images():
    global features, labels
    img_size = 200
    bee_label = 0
    hornet_label = 1
    bee_path = path + 'bbees'
    hornet_path = path + 'hornets'
    training_data = []

    bee_count = 0
    for bimg in os.listdir(bee_path):
        img_arr = cv2.imread(os.path.join(bee_path, bimg), cv2.IMREAD_COLOR)
        img_arr = cv2.resize(img_arr, (img_size, img_size))
        training_data.append([img_arr, bee_label])

        bee_count += 1
        if bee_count > 351:
            break

    for himg in os.listdir(hornet_path):
        img_arr = cv2.imread(os.path.join(hornet_path, himg), cv2.IMREAD_COLOR)
        img_arr = cv2.resize(img_arr, (img_size, img_size))
        training_data.append([img_arr, hornet_label])

    #shuffle data
    random.seed(seed)
    random.shuffle(training_data)

    #seperate features and labels
    for feat, lab in training_data:
        features.append(feat)
        labels.append(lab)
    features = np.array(features).reshape(-1, img_size, img_size, 3)
    labels = np.array(labels)

    #save data
    while True:
        answer = input('Would you like to save features and labels? [Y/N] ')
        if answer == 'Y':
            save_processed_data(features, labels)
            break
        elif answer == 'N':
            break

def save_processed_data(feat, lab):
    #save features
    pickle_out = open(path + 'features_pickle', 'wb')
    pickle.dump(feat, pickle_out)
    pickle_out.close()

    #save labels
    pickle_out = open(path + 'labels_pickle', 'wb')
    pickle.dump(lab, pickle_out)
    pickle_out.close()

    print('Data saved!')

read_images()

def load_processed_data():
    global features, labels
    try:
        features = pickle.load(open(path + 'features_pickle', 'rb'))
        labels = pickle.load(open(path + 'labels_pickle', 'rb'))
    except Exception as e:
        print("Unable to load data: " + str(e))
        return False
    print("Preprocessed Data Loaded!")
    return True

load_processed_data()

#Hypertuning #Define the model as a function

def model(x_train, y_train, x_val, y_val, params):
    in_layer = layers.Input((200, 200, 3))
    conv1 = layers.Conv2D(filters=32, kernel_size=5, padding='same', activation=params['activation'])(in_layer)
    pool1 = layers.MaxPool2D()(conv1)
    conv2 = layers.Conv2D(filters=64, kernel_size=5, padding='same', activation=params['activation'])(pool1)
    pool2 = layers.MaxPool2D()(conv2)
    flatten = layers.Flatten()(pool2)
    dense1 = layers.Dense(params['hidden_layers'], activation=params['activation'])(flatten)
    dropout1 = layers.Dropout(params['dropout'])(dense1)
    preds = layers.Dense(2, params['activation'])(dropout1)

    model = Model(in_layer, preds)
    model.compile(loss=params['losses'],
                  optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                  metrics=["acc"])

    history = model.fit(x_train, y_train,
                        validation_data=(x_val, y_val),
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        verbose=1)
    return history, model

p = {'lr': (0.5, 5, 10),
     'first_neuron': [4, 16, 32],
     'hidden_layers': [0, 1, 2],
     'batch_size': (2, 30, 10),
     'epochs': [15],
     'dropout': (0, 0.5, 5),
     'optimizer': [Adam, Nadam, RMSprop],
     'losses': [logcosh, binary_crossentropy],
     'activation':[relu, elu],
     'last_activation': [sigmoid]}

t = ta.Scan(x=features,
            y=labels,
            model=model,
            fraction_limit=0.01,
            params=p,
            experiment_name="1")







