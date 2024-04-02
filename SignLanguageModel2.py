#!/usr/bin/env python
# coding: utf-8

# # Sign_language_detecttion_Model 

# #### Importing the data from the folder and checking classes
# 

# In[20]:


import sys

# Check if the user provided a command-line argument
if len(sys.argv) < 2:
    print("Please provide a file path as a command-line argument.")
    sys.exit(1)

# Get the file path from the command-line arguments
file_path = sys.argv[1]

# Now you can use file_path in your code


# #### Visualising some of the pictures 

# In[23]:


path=file_path
print(file_path)


# #### Creating a variable for main folder  

# In[25]:


from os import listdir 
root_dir = listdir(path)
image_list, label_list = [], []
print(root_dir)


# #### resizing our images 

# In[4]:


from keras.preprocessing import image
from PIL import Image
from tensorflow.keras.utils import img_to_array
for directory in root_dir:
    i=0
    for files in listdir(f"{path}/{directory}"):
        image_path = f"{path}/{directory}/{files}"
        image = Image.open(image_path)
        image = image.resize((50,50))
        image = img_to_array(image)
        image_list.append(image)
        
        label_list.append(directory)
        i+=1
        if i>3000:
          break


# In[5]:


import pandas as pd

print(directory)
label_counts = pd.DataFrame(label_list).value_counts()
sorted(label_counts)


# In[6]:


num_classes = len(label_counts)
num_classes


# In[8]:


import numpy as np


# In[9]:


label_list = np.array(label_list)
label_list.shape


# #### Starting our training phase of the model by splitting the data 

# In[10]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(image_list, label_list, test_size=0.2, random_state = 10) 


# In[11]:


x_train = np.array(x_train, dtype=np.float16) / 255.0
x_test = np.array(x_test, dtype=np.float16) / 255.0
x_train = x_train.reshape( -1, 50,50,3)
x_test = x_test.reshape( -1, 50,50,3)


# #### preparing labels for the classes present
# 

# In[12]:


from sklearn.preprocessing import  LabelBinarizer
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)
print(lb.classes_)


# In[13]:


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.3)


# In[14]:


print(y_train.shape)
print(x_train.shape)


# #### Importing libraries for the model 

# In[15]:


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, LeakyReLU


# In[16]:


model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same",input_shape=(50,50,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(90, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(90, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(60, activation="relu"))
model.add(Flatten())

model.add(Dense(40, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))
model.summary()


# #### Compiling the model 

# In[17]:


from tensorflow import keras
early_stop = keras.callbacks.EarlyStopping(monitor='val-loss', patience=3)
model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size = 32, epochs =10,validation_data=(x_val,y_val),callbacks=[early_stop])


# #### Evaluation of the trained model 

# In[18]:


scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")


# ##### Saving the trained model 

# In[19]:


model.save('sign_lang_model3.h5')


# #### opening model
# 

# In[30]:


import tensorflow as tf


# In[31]:


loaded_model = tf.keras.models.load_model('sign_lang_model2.h5')


# In[32]:


from keras.preprocessing import image
from PIL import Image
from tensorflow.keras.utils import img_to_array
import matplotlib.pyplot as plt


# ##### Entering the path of the image to be tested on 

# In[33]:


import imutils
import tkinter as tk
from tkinter import filedialog


# In[34]:


image = Image.open("D:/main_data/asl_alphabet_test/nothing_test.jpg")
plt.imshow(image)


# In[35]:


from PIL import Image,ImageOps
import matplotlib.pyplot as plt

import numpy as np
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
import os
import h5py


# #### Predicting the sign of the image put into the model 

# In[39]:


shape = ((50,50,3))
model = loaded_model
test_image = image.resize((80,80))
test_image = preprocessing.image.img_to_array(test_image)
test_image = test_image / 255
test_image = np.expand_dims(test_image, axis =0)
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
predictions = model.predict(test_image)
scores = tf.nn.softmax(predictions[0])
scores = scores.numpy()
image_class = class_names[np.argmax(scores)]
print(image_class)


# In[ ]:




