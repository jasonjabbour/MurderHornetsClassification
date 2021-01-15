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


try: 
    import lime
except:
    sys.path.append(os.path.join('..','..')) # add the currect directory
    import lime
from lime import lime_image

from skimage.segmentation import mark_boundaries

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.replace('\\','/')

#where image for prediction is located
image_path = dir_path+'/image_for_prediction'
image_size = 200

categories = ['Bumble Bee','Murder Hornet']

image_exists = False
#read in the image and proccess it
for img in os.listdir(image_path):
    image_array = cv2.imread(os.path.join(image_path,img),cv2.IMREAD_COLOR)
    image_array = cv2.resize(image_array,(image_size,image_size))
    image_array = np.array(image_array).reshape(-1,image_size,image_size,3)

    image_exists = True
    #only read the first image
    break

if not image_exists:
    print('No image found in folder: image_for_prediction \nPlease add an image to the folder and try again')
    sys.exit()

#try to load the following saved model
try:
    loaded_model = tf.keras.models.load_model(dir_path+'/saved_Models/'+'Saved-BeeHornet-3-conv-128-layer_size-1-dense_layer-1610671724-Final')
except:
    print("Saved model does not exist")
    sys.exit()

# #make prediction
# prediction = loaded_model.predict([image_array])
# #format prediciton 0-bee, 1-murder hornet
# image_class =categories[int(prediction[0][0])]

#Start Lime Analysis
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(image_array.astype('double'),loaded_model.predict)

#see the features
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0])
plt.imshow(mark_boundaries/2+.5,mask)


