#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 14:12:55 2018

@author: emblab
"""


import numpy as np
np.random.seed(2017)

import os
import time
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input
from keras.models import Model
from keras.models import load_model
import tkinter as tk
from tkinter import filedialog

dict={0:'No DR',1: 'Mild',2 : 'Moderate',3:'Severe',4: 'Proliferative DR'}

model=load_model('/home/priyank/Desktop/healthcare/healthcare_resnet50_30_0.h5')


#img_path='/home/emblab/Healthcare/test_healthcare/19380_right.jpeg'



root = tk.Tk()
root.withdraw()
img_path = filedialog.askopenfilename()


img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print('Input image shape:', x.shape)

pr=model.predict(x)
pr1=np.argmax(pr)
print('\n\n\nResult=',dict[pr1])