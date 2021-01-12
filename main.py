import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import sys

#get directory path
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.replace('\\','/')
seed = 1122021
features = []
labels = []

def read_images():
    '''
        Read the dataset of images. Murder Hornets and Bees. 
    '''
    global features, labels

    image_size = 200
    bee_label = 0 
    hornet_label = 1

    #the paths to the data
    path_lst = ['Bees/train','MurderHornets']
    #create path
    path_bees = dir_path + '/Data/' + path_lst[0]
    path_hornets = dir_path + '/Data/' + path_lst[1]

    training_data = []
    bee_img_count = 1
    #read in all the bees images
    for bee_img in os.listdir(path_bees):
        #for each bee_img name, join it to the path of bees to read it
        image_array = cv2.imread(os.path.join(path_bees,bee_img),cv2.IMREAD_COLOR)
        #resize image from (200,200,3)...3 represents colors
        image_array = cv2.resize(image_array, (image_size,image_size))
        #add image and its label
        training_data.append([image_array,bee_label])

        #only use first 352 images
        bee_img_count+=1
        if bee_img_count > 352:
            break

            
    #read in all the hornets images
    for horn_img in os.listdir(path_hornets):
        #for each horn_img name, join it to the path of bees to read it
        image_array = cv2.imread(os.path.join(path_hornets,horn_img),cv2.IMREAD_COLOR)
        #resize image
        image_array = cv2.resize(image_array, (image_size,image_size))
        #add image and its label
        training_data.append([image_array,hornet_label])

    #shuffle data
    random.seed(seed)
    random.shuffle(training_data)

    #separate features and labels 
    features = []
    labels = []
    for feat,lab in training_data:
        features.append(feat)
        labels.append(lab)
  
    #change type...for kares array must be an np array
    features = np.array(features).reshape(-1,image_size,image_size,3) #-1 for all features, 3 for all three colors

    #save data?
    while True:
        answer = input('Would you like to save features and labels?[Y/N] ')
        if answer == 'Y':
            #call the save function
            save_processed_data(features,labels)
            break
        elif answer == 'N':
            break

def save_processed_data(features,labels):
    '''
        save the features and the labels
    ''' 

    #save features
    pickle_out = open(dir_path + '/features_pickle','wb')
    pickle.dump(features,pickle_out)
    pickle_out.close()

    #save labels 
    pickle_out = open(dir_path + '/labels_pickle','wb')
    pickle.dump(labels,pickle_out)
    pickle_out.close()

    print('Data saved!')

def load_processed_data():
    '''
        Load features and lables.

        Return True/False if successful
    '''
    global features, labels
    try:
        features = pickle.load(open(dir_path + '/features_pickle','rb'))
        labels = pickle.load(open(dir_path + '/labels_pickle','rb'))
    except Exception as e:
        print("Unable to load data: " + str(e))
        return False
    
    print("Preprocessed Data Loaded!")
    return True



#driver
if __name__ == '__main__':

    exitloop1 = False

    while True:
        #process new data?
        answer2 = input('Would you like to read new images?[Y/N] ')
        if answer2 == 'Y':
            #read in the image dataset
            read_images()
            break
        elif answer2 == 'N':
            #read existing data?
            while True:
                answer2_5 = input('Would you like to load existing processed data?[Y/N] ')
                if answer2_5 == 'Y':
                    #load the preprocessed data
                    load_success = load_processed_data()
                    #if not success ask again
                    if load_success:
                        exitloop1 = True #to exit outer loop 
                        break
                elif answer2_5 == 'N':
                    exitloop1 = True #to exit outer loop
                    break

        if exitloop1:
            break
    
    #run model?
    while True:
        answer3 = input('Would you like to run a CNN?[Y/N] ')
        if answer3 == 'Y':
            #...
            break 
        elif answer3 == 'N':
            break


#print(image_array.shape)
# plt.imshow(image_arry)
# plt.show()
