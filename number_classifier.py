import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

NEEDS_TRAIN=True
EPOCHS_NUM=3

if NEEDS_TRAIN==True:
    mnist=tf.keras.datasets.mnist
    #split to training and testing data respectively.
    (x_train,y_train), (x_test,y_test) = mnist.load_data() #X: pixel, Y: classification

    #Scale down to between 0.0 and 1.0:
    x_train=tf.keras.utils.normalize(x_train,axis=1)
    x_test = tf.keras.utils.normalize(x_test,axis=1)

    #Create model
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28,28))) #28x28 image size, flattens the data to an one dim. data structure.

    model.add(tf.keras.layers.Dense(128, activation='relu')) #Rectified linear unit.

    model.add(tf.keras.layers.Dense(128, activation='relu')) #Second inner layer

    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train,y_train, epochs=EPOCHS_NUM) #Fit and train the model

    #If trained
    model.save('handwritten.model')
else:
    model=tf.keras.models.load_model('handwritten.model')
    
    #loss,accuracy = model.evaluate(x_test,y_test)
    #print(loss)
    #print(accuracy)
    
    file_path= "./needs_classified/3.png"
    try:
        img = cv2.imread(file_path)[:,:,0]
        
        prediction = model.predict(img)
        
        print("This digit is classified as:"+str(np.argmax(prediction)))
        
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error, the sizes of the data were incorrect or probably missing.")
        
        
    
    