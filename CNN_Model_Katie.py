#imports
import cv2
import numpy as np
import os

from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

import tensorflow as tf

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.replace('\\','/')


#labels = ['bbees', 'hornets']
labels = ['Bees','MurderHornets']
img_size = 200

def seperate_data(root):
    global labels
    for label in labels:
        path = root + label
        count = 0
        for img in os.listdir(path):
            if count <= 70:
                folder_name = 'test'
            elif (count > 70) & (count <= 352):
                folder_name = 'train'
            else:
                break
            os.makedirs(path + '/' + folder_name, exist_ok=True)
            source = os.path.join(path, img)
            destination = path + '/' + folder_name + '/' + img
            os.rename(source, destination)
            count += 1

#seperate_data('/Users/katiekorn/PycharmProjects/ds/bees/data/')
#seperate_data(dir_path+'/Data/')


def get_data(root):
    global labels, img_size
    data = []
    for label in labels:
        path = os.path.join(root, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                img_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([img_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data, dtype=object)

val = get_data('/Users/katiekorn/PycharmProjects/ds/bees/data/test')
train = get_data('/Users/katiekorn/PycharmProjects/ds/bees/data/train')

# val = get_data(dir_path+'/Data/Bees/test')
# train = get_data('/Users/katiekorn/PycharmProjects/ds/bees/data/train')


#Data Processing
x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
  x_train.append(feature)
  y_train.append(label)

for feature, label in val:
  x_val.append(feature)
  y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

#Augment training data
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(x_train)

# #Generate sample images
# for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=16):
#     for i in range(16):
#         plt.subplot(330+1+i)
#         plt.imshow(x_batch[i])
#     plt.show()
#     break

#Define the model
model = Sequential()
model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(200,200,3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(2, activation="softmax"))

model.summary()

#Compile the model
opt = Adam(lr=0.000001)
model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])

#Train the model
batch_size = 64
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=(x_train.shape[0]//batch_size), epochs=15, validation_data=(x_val, y_val))

#Evaluate the model
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(15)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#Classification Report
predictions = np.argmax(model.predict(x_val), axis=-1)
predictions = predictions.reshape(1,-1)[0]
print(classification_report(y_val, predictions, target_names = ['Bee (Class 0)','Hornet (Class 1)'], labels=y_val))







