import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time
import matplotlib.pyplot as plt
import keras
from keras.preprocessing import image
from keras.applications import inception_v3 as inc_net
from CNN_Model import predict_image
try: 
    import lime
except:
    sys.path.append(os.path.join('..','..')) # add the currect directory
    import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries

#get path
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.replace('\\','/')

#where is image for prediction is located
image_path = dir_path+'/image_for_prediction'
image_size = 200

categories = ['Bumble Bee','Murder Hornet']

def set_image_size(size):
    '''
        image size setter
    '''
    global image_size
    image_size = size

def read_image():
    '''
        Read the image that is being fed in from the folder: image_for_prediction
        Image must be named: feedme
        Return the processed image
    '''
    image_exists = False
    #read in the image and proccess it
    for img in os.listdir(image_path):
        
        #only use img named feed me
        if "feedme" in img:
            out = []
            img = image.load_img(os.path.join(image_path,img), target_size=(image_size,image_size))
            x = image.img_to_array(img)
            x = np.expand_dims(x,axis=0)
            x = inc_net.preprocess_input(x)
            out.append(x)
            image_array = np.vstack(out)

            image_exists = True
            #only read the first image
            break

    if not image_exists:
        print('No image found in folder: image_for_prediction  named "feedme"\nPlease add an image to the folder and try again')
        sys.exit()
    
    return image_array

def load_model(our_own_model=True):
    '''
        Read in a model. 
        Either our CNN model that was trained or the inception pre-trained model 
        return the loaded model
    '''

    #Names of saved model in the saved_Models folder
    model_names = ['Saved-BeeHornet-3-conv-128-layer_size-1-dense_layer-1610671724-Final',
                    'BeeClassificationDatainceptionmodel']

    which_model = 0 
    if not our_own_model:
        which_model = 1

    #try to load the following saved model
    try:
        loaded_model = tf.keras.models.load_model(dir_path+'/saved_Models/'+model_names[which_model])
        return loaded_model
    except:
        print("Saved model does not exist in the folder: saved_Models")
        sys.exit()

def start_lime(image_array,loaded_model):
    '''
        Start the lime analysis
        return the explanation object
    '''

    #Start Lime Analysis
    explainer = lime_image.LimeImageExplainer()
    #required the image and model
    explanation = explainer.explain_instance(image_array[0].astype('double'),loaded_model.predict)

    return explanation

def view_analysis(explanation):
    '''
        Plot the image and the features detected on the image
    '''
    # # see the superpixels that the model found to be most relevant to make a prediciton
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[np.array(0)],positive_only=True,hide_rest=True)
    plt.imshow(mark_boundaries(temp/2+0.5,mask))
    plt.show()

    heat_map_plot = False
    
    #heat map -------------------------------------------
    if heat_map_plot:
        #Select the same class explained on the figures above.
        ind =  explanation.top_labels[0]

        #Map each explanation weight to the corresponding superpixel
        dict_heatmap = dict(explanation.local_exp[ind])
        heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 

        #Plot. The visualization makes more sense if a symmetrical colorbar is used.
        plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
        plt.colorbar()
        plt.show()

#driver
if __name__ == '__main__':


    while True:
        answer1 = input('Would you like to use regular CNN Model or Pre-trained InceptionV3 Model?[cnn/inc] ')

        print('Reading image named "feedme" from folder: image_for_prediction')

        if answer1 == 'cnn':
            #set the image size first
            set_image_size(200)
            #read the image and return processed image
            image_array = read_image()
            #use regular keras model
            loaded_model = load_model()
            break
        elif answer1 == 'inc':
            #set the image size first
            set_image_size(150)
            #read the image and return processed image
            image_array = read_image()
            #use inception model
            loaded_model = load_model(False)
            break

    #start lime analysis return explanation object
    exp = start_lime(image_array,loaded_model)

    #view features
    view_analysis(exp)


