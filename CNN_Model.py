import numpy as np
import cv2
import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

#model = Sequential()


def create_CNN_model(features, labels):
    '''
        Start CNN
    '''

    #normalize data. Scale pixel data (max is 255)
    features = features/255.0

    #using sequential model
    model = Sequential()

    #layer 1. Convolution layer. 64 units. 3x3 window size
    model.add(Conv2D(64,(3,3),input_shape=features.shape[1:]))
    #Add activation layer (rectified linear)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #layer 2
    model.add(Conv2D(64,(3,3)))
    #Add activation layer (rectified linear)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #layer 3. must flatten into 1D first
    model.add(Flatten())
    model.add(Dense(64))

    #output layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

    model.fit(features,labels,batch_size=32,validation_split=.1)
    return 

