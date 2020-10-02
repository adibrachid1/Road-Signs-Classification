#
# Currently, I am testing on the train folder (later on will include the test folder)
# Using tensorflow library to build our model
# 1- Run the split_folder.py script which will be in the same folder containing the train folder data
# this will create 3 subdirectories in a folder which you will name data (train - val - test)
# Currently data is not so good bcz repeated and taken from video - should be improved later on
# 2- run this script run.py to prepare the data
# then to create the model (not yet implemented)


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2



batch_size = 128 #to be modified later on
epochs = 15 #to be modified later on
IMG_HEIGHT = 200
IMG_WIDTH = 200

#for the generator we can do some augmentation regarding the shifting later on
train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

#load images, apply data generator, resize
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

#return batch from dataset in fomr os (x_train,y_train) <=> (features,labels)
sample_training_images, y = next(train_data_gen)
#print(y[:5])
#plotImages(sample_training_images[:5])

#So far, data is preprocessed and now we need to create the model for the NN

# Create the model and add layers
# Nomber of layers, activations and shapes will be fixed later
model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation=’relu’, input_shape=None))
model.add(Conv2D(64, kernel_size=3, activation=’relu’, input_shape=None))
model.add(Conv2D(64, kernel_size=3, activation=’relu’, input_shape=None))
model.add(Conv2D(64, kernel_size=3, activation=’relu’, input_shape=None))
model.add(Flatten())
model.add(Dense(43, activation=’softmax’))