# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:50:58 2024

@author: arati
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
#image data generator is the package to label the image and it will automatically lable all the image
img = image.load_img(r"D:\Python tasks_Ds\Sentiment Detection\Training\Happy\IMG_20240308_124202.jpg")
plt.imshow(img)

i1 = cv2.imread("D:\Python tasks_Ds\Sentiment Detection\Training\Happy\IMG_20240308_124202.jpg")
i1
# 3 dimension metrics are created for the image
# the value ranges from 0-255

i1.shape
train = ImageDataGenerator(rescale = 1/200)
validation = ImageDataGenerator(rescale=1/200)
# to scale all the images i need to divide woth 255
#we need to resize the image using 200, 200 pixel

train_dir = r"D:\Python tasks_Ds\Sentiment Detection\Training"
validation_dir = r"D:\Python tasks_Ds\Sentiment Detection\Validation"

train_dataset = train.flow_from_directory(train_dir,
                                          target_size=(200, 200),  # Resize images to 200x200 pixels
                                          batch_size=32,
                                          class_mode='binary')  # Binary classification

validation_dataset = validation.flow_from_directory(validation_dir,
                                                    target_size=(200, 200),
                                                    batch_size=32,
                                                    class_mode='binary')


train_dataset.class_indices
train_dataset.classes
validation_dataset.classes


model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation = 'relu',input_shape = (200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2), #3 filter we applied here
                                    #
                                    tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    tf.keras.layers.Flatten(),
                                    #
                                    tf.keras.layers.Dense(512,activation='relu'),
                                    #
                                    tf.keras.layers.Dense(1,activation='sigmoid')
                                    ])

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),  # Corrected typo
    metrics=['accuracy']
)

model_fit = model.fit(train_dataset,epochs=8)

dir_path = r"D:\Python tasks_Ds\Sentiment Detection\Training\Happy"
for i in os.listdir(dir_path) :
    print(i)
    img = image.load_img(dir_path+ '//'+i, target_size = (200,200))
    plt.imshow(img)
    plt.show()
    
dir_path = r"D:\Python tasks_Ds\Sentiment Detection\Training\Happy"
for i in os.listdir(dir_path) :
    print(i)
    img = image.load_img(dir_path+ '//'+i, target_size = (200,200))
    plt.imshow(img)
    plt.show()

    
    x= image.img_to_array(img)
    x=np.expand_dims(x,axis = 0)
    images = np.vstack([x])
    
    val = model.predict(images)
    if val == 0 :
        print('You Seems Happy')
    else:
        print('You Seems Sad')
    


import pickle

# Save the trained model using pickle
with open("sentiment_model.pkl", "wb") as file:
    pickle.dump(model, file)















