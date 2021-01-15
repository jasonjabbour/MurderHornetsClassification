import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import sys
import time
import seaborn
import pandas as pd
from CNN_Model import create_CNN_model, predict_image

#get directory path
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.replace('\\','/')
seed = 1122021
features = []
labels = []
training_data = []
image_size = 200

#plotting variables
bees_count = 0 
hornet_count = 0 


def read_images():
    '''
        Read the dataset of images. Murder Hornets and Bees. 

        Return True if successful 
    '''
    global features, labels, training_data, bees_count, hornet_count, image_size

    bee_label = 0 
    hornet_label = 1

    #the paths to the data
    path_lst = ['Bees','MurderHornets']
    #create path
    path_bees = dir_path + '/Data/' + path_lst[0]
    path_hornets = dir_path + '/Data/' + path_lst[1]

    #count number of images
    bees_count = 0
    #read in all the bees images
    for bee_img in os.listdir(path_bees):
        #for each bee_img name, join it to the path of bees to read it
        image_array = cv2.imread(os.path.join(path_bees,bee_img),cv2.IMREAD_COLOR)
        #resize image from (200,200,3)...3 represents colors
        image_array = cv2.resize(image_array, (image_size,image_size))
        #add image and its label
        training_data.append([image_array,bee_label])

        #rotate image 3 different times
        image_arrayRot90 = cv2.rotate(image_array,cv2.ROTATE_90_CLOCKWISE)
        image_arrayRot180 = cv2.rotate(image_array,cv2.ROTATE_180)
        image_arrayRot270 = cv2.rotate(image_array,cv2.ROTATE_90_COUNTERCLOCKWISE)

        #add rotated images
        training_data.append([image_arrayRot90,bee_label])
        training_data.append([image_arrayRot180,bee_label])
        training_data.append([image_arrayRot270,bee_label])

        #only use first 352 images
        bees_count+=4

        if bees_count >= 352*4:
            break

    #count amount of images 
    hornet_count = 0 
    #read in all the hornets images
    for horn_img in os.listdir(path_hornets):
        #for each horn_img name, join it to the path of bees to read it
        image_array = cv2.imread(os.path.join(path_hornets,horn_img),cv2.IMREAD_COLOR)
        #resize image
        image_array = cv2.resize(image_array, (image_size,image_size))
        #add image and its label
        training_data.append([image_array,hornet_label])

        #rotate image 3 different times
        image_arrayRot90 = cv2.rotate(image_array,cv2.ROTATE_90_CLOCKWISE)
        image_arrayRot180 = cv2.rotate(image_array,cv2.ROTATE_180)
        image_arrayRot270 = cv2.rotate(image_array,cv2.ROTATE_90_COUNTERCLOCKWISE)

        #add rotated images
        training_data.append([image_arrayRot90,hornet_label])
        training_data.append([image_arrayRot180,hornet_label])
        training_data.append([image_arrayRot270,hornet_label])

        hornet_count+=4

    #shuffle data
    random.seed(seed)
    random.shuffle(training_data)

    #separate features and labels 
    for feat,lab in training_data:
        features.append(feat)
        labels.append(lab)
  

    #change type...for kares array must be an np array
    features = np.array(features).reshape(-1,image_size,image_size,3) #-1 for all features, 3 for all three colors
    labels = np.array(labels)

    #save data?
    while True:
        answer = input('Would you like to save features and labels?[Y/N] ')
        if answer == 'Y':
            #call the save function
            save_processed_data(features,labels)
            break
        elif answer == 'N':
            break

    return True

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


def plot_graphics():
    '''
        Helper function to plot data
    '''

    #bar plot
    count1 = bees_count #got from for loop in main.py
    count2 = hornet_count #got from for loop in main.py
    objects = ['Bees','Murder Hornets']
    thecounts = [count1,count2]
    df = pd.DataFrame({"Class":objects, "Number of Images":thecounts})

    plt.figure(figsize=(8, 6))
    splot=seaborn.barplot(x="Class",y="Number of Images",data=df)
    for p in splot.patches:
        splot.annotate(format(p.get_height(), '.0f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    size=15,
                    xytext = (0, -12), 
                    textcoords = 'offset points')
    plt.xlabel("Class", size=14)
    plt.ylabel("Number of Images", size=14)
    plt.title('Class Division of Bees and Murder Hornet Images after Data Augmentation')
    plt.ylim(0,1500)
    plt.show()

#driver
if __name__ == '__main__':

    plot_data = False

    exitloop1 = False
    data_ready = False

    while True:
        #process new data?
        answer2 = input('Would you like to read new images?[Y/N] ')
        if answer2 == 'Y':
            #read in the image dataset
            read_images()
            data_ready = True
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
                        data_ready = True
                        exitloop1 = True #to exit outer loop 
                        break
                elif answer2_5 == 'N':
                    exitloop1 = True #to exit outer loop
                    break

        if exitloop1:
            break
    
    #run model?
    while True:
        answer3 = input('Would you like to create a CNN Model?[Y/N] ')
        if answer3 == 'Y':
            if not data_ready:
                print('No pre-processed Data Available. Terminating...')
                break
            
            #create new model
            create_CNN_model(features,labels)

            break 
        elif answer3 == 'N':
            break

    while True:
        answer4 = input('Would you like to used saved model to predict new image?[Y/N] ')
        if answer4 == 'Y':
            print('Locating image in folder: image_for_prediction...')
            predict_image(image_size)
            break
        elif answer4 == 'N':
            break


    #only if manually specified
    if plot_data:
        plot_graphics()

        
#print(image_array.shape)
# plt.imshow(image_arry)
# plt.show()
