from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import load_img,img_to_array

image=image.load_img(r"D:\Python tasks_Ds\Sentiment Detection\Training\Happy\IMG_20240308_124202.jpg")
plt.imshow(image)
plt.show()

i1=cv2.imread(r"D:\Python tasks_Ds\Sentiment Detection\Training\Happy\IMG_20240308_124202.jpg")
print(i1) # convert to ND array



train=ImageDataGenerator(rescale=1/200)
validation=ImageDataGenerator(rescale=1/200)

train_data=train.flow_from_directory(r"D:\Python tasks_Ds\Sentiment Detection\Training",target_size=(200,200),
                                    batch_size=32,
                                    class_mode="binary")
validation_data=validation.flow_from_directory(r"D:\Python tasks_Ds\Sentiment Detection\Validation",target_size=(200,200),
                                    batch_size=32,
                                    class_mode="binary")

train_data.class_indices

train_data.classes

model=tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation="relu",input_shape=(200,200,3)),
                            tf.keras.layers.MaxPool2D(2,2),
                            tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
                            tf.keras.layers.MaxPool2D(2,2),

                            tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
                            tf.keras.layers.MaxPool2D(2,2),

                            tf.keras.layers.Flatten(),

                            tf.keras.layers.Dense(512,activation="relu"),
                            tf.keras.layers.Dense(1,activation="sigmoid")
                           ])

model.compile(loss="binary_crossentropy",
             optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
             metrics=["accuracy"])

model_fit=model.fit(train_data,epochs=8)

dir_path=r"D:\Python tasks_Ds\Sentiment Detection\Testing"
for i in os.listdir(dir_path):
    print(i)
    
dir_path=r"D:\Python tasks_Ds\Sentiment Detection\Testing"
for i in os.listdir(dir_path):
    img=load_img(dir_path+"//"+i,target_size=(200,200))
    plt.imshow(img)
    plt.show()
    
    
 
dirpath=r"D:\Python tasks_Ds\Sentiment Detection\Testing"
   
for i in os.listdir(dirpath):
    img=load_img(dirpath+"//"+i,target_size=(200,200))
    plt.imshow(img)
    plt.show()
    
    x=img_to_array(img)
    x=np.expand_dims(x,axis=0)
    images=np.vstack([x])
    val=model.predict(images)
    if val==0:
        print("Happy")
    else:
        print("not happy")
        
model.save("emotion_detector_model.h5")
   