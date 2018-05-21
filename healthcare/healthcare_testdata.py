#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 14:16:08 2018

@author: priyank
"""


import numpy as np
np.random.seed(2017)

import os
import time
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.models import load_model
from numpy import genfromtxt



dict={0:'No DR',1: 'Mild',2 : 'Moderate',3:'Severe',4: 'Proliferative DR'}
trainLabelscsv = genfromtxt('/home/priyank/Desktop/healthcare/trainLabels.csv',dtype=str, delimiter=',')



PATH='/media/priyank/6442175942172F741/test_healthcare'
files=os.listdir(PATH)
img_data_list=[]
labels=[]


#files.sort(key=int)
count=0
y=trainLabelscsv.shape[0]
for i in files:
       print(i)
       img_path = PATH+ '/'+ i 
       print('count=',count)
       count=count+1
       img = image.load_img(img_path, target_size=(224, 224))
       x = image.img_to_array(img)
       x = np.expand_dims(x, axis=0)
       x = preprocess_input(x)
       print('Input image shape:', x.shape)
       img_data_list.append(x)
       for j in range(0,y):
           if i==trainLabelscsv[j][0]+'.jpeg':
                  print(trainLabelscsv[j][1])
                  labels.append(trainLabelscsv[j][1])
                  
                  

img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)
num_of_samples = img_data.shape[0]


                  
                  
#labels
num_classes=5
Y = np_utils.to_categorical(labels, num_classes)


#x,y = shuffle(img_data,Y, random_state=2)

#X_test, y_test = shuffle(img_data,Y, random_state=2)


X_test, y_test=img_data,Y
model=load_model('/home/priyank/Desktop/healthcare/healthcare_resnet50_30_0.h5')

(loss, accuracy) = model.evaluate(X_test, y_test, batch_size=10, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))



pr=model.predict(X_test)
pr1=np.zeros((pr.shape[0],1),dtype='int64')
for i in range(pr.shape[0]):
  pr1[i][0]=np.argmax(pr[i])
  #print(dict[pr1[i][0]], i)

#vec=model.predict(X_test)

