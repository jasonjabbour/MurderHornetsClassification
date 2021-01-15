import numpy as np
import cv2
import os 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.replace('\\','/')

def create_CNN_model(features, labels):
    '''
        Start Convolutional Neural Network
    '''

    #normalize data. Scale pixel data (max is 255)
    features = features/255.0

    feat_shape = features.shape[1:]

    #dense_layers = [0,1,2]
    dense_layers = [1]

    #layer_sizes = [32,64,128]
    layer_sizes = [128]

    #convolutional_layers = [1,2,3]
    convolutional_layers = [3]

    set_epochs = 8

    for d_l in dense_layers:
        for l_s in layer_sizes:
            for c_l in convolutional_layers:

                #save each model
                MODEL_NAME = 'BeeHornet-{}-conv-{}-layer_size-{}-dense_layer-{}-Final'.format(c_l,l_s,d_l,int(time.time()))
                tensorboard = TensorBoard(log_dir='logs/{}'.format(MODEL_NAME)) 
                #activate tensorflow tensorboard --logdir=logs/

                #using sequential model
                model = Sequential()

                #First convolutional layer must have input size. layer 1. Convolution layer. 64 units. 3x3 window size
                model.add(Conv2D(l_s,(2,2),input_shape=feat_shape))
                #Add activation layer (rectified linear)
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2,2)))

                #try x number convolutional layers
                for x in range(c_l-1): #-1 because we already have first conv layer above
                    model.add(Conv2D(l_s,(2,2)))
                    #Add activation layer (rectified linear)
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2,2)))

                #first dense layer must have flatten. Must flatten into 1D
                model.add(Flatten())

                #try x number of dense layers
                for y in range(d_l):
                    #l_s will be the dimensionality of the outut space
                    model.add(Dense(l_s))
                    model.add(Activation('relu'))

                #output layer
                model.add(Dense(1))
                model.add(Activation('sigmoid'))

                model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

                model.fit(features,labels,batch_size=32,epochs=set_epochs,validation_split=.1, callbacks=[tensorboard])

                model.save(dir_path+'/saved_Models/Saved-'+MODEL_NAME)


def predict_image(image_size):
    '''
        Image must be named "feedme"
        Read first image from the predict folder with the name "feedme" and classify it
        Return the 0 or 1, 0 for be, 1 for Murder hornet
    '''

    categories = ['Bumble Bee','Murder Hornet']

    #where image for prediction is located
    image_path = dir_path+'/image_for_prediction'

    image_exists = False
    #read in the image and proccess it
    for img in os.listdir(image_path):

        if "feedme" in img:
            image_array = cv2.imread(os.path.join(image_path,img),cv2.IMREAD_COLOR)
            image_array = cv2.resize(image_array,(image_size,image_size))
            image_array = np.array(image_array).reshape(-1,image_size,image_size,3)

            image_exists = True
            #only read the first image
            break

    if not image_exists:
        print('No image found in folder: image_for_prediction  named "feedme"\nPlease add an image to the folder and try again')
        return

    #try to load the following saved model
    try:
        loaded_model = tf.keras.models.load_model(dir_path+'/saved_Models/'+'Saved-BeeHornet-3-conv-128-layer_size-1-dense_layer-1610671724-Final')
    except:
        print("Saved model does not exist")

    #make prediction
    prediction = loaded_model.predict([image_array])
    #format prediciton 0-bee, 1-murder hornet
    image_class =categories[int(prediction[0][0])]
    
    print('-'*40,'\n','Image class:', image_class,'\n'+'-'*40)
    return int(prediction[0][0])

    

