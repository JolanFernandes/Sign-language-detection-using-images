#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import imutils
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.activations import softmax
import numpy as np
from PIL import Image,ImageTk


# In[2]:


model = tf.keras.models.load_model('sign_lang_model2.h5')


# In[3]:


from tkinter import Label


# In[4]:


def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    
    if file_path:
        image=cv2.imread(file_path)
        pil_image = Image.open(file_path)
        shape = ((80,80,3))
        model = tf.keras.models.load_model('sign_lang_model2.h5')
        test_image = pil_image.resize((80,80))
        img=pil_image.resize((250,200))
        photo = ImageTk.PhotoImage(img)
        image_label.config(image=photo)
        image_label.image = photo
        test_image = preprocessing.image.img_to_array(test_image)
        test_image = test_image / 255
        test_image = np.expand_dims(test_image, axis =0)
        class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
        predictions = model.predict(test_image)
        scores = tf.nn.softmax(predictions[0])
        scores = scores.numpy()
        image_class = class_names[np.argmax(scores)]
        
        val=print(image_class)
        label.configure(foreground="#011638",text="predicted sign is :"+image_class)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        


# In[5]:


root = tk.Tk()
root.title("Sign Language Recognition")
open_button = tk.Button(root, text="Select Image", command=open_image)
open_button.pack(side='bottom',pady=50)
root.geometry("550x450")
label=Label(root,background='#CDCDCD',font=('arial',15,'bold'))
image_label = tk.Label(root)
image_label.pack()
label.pack(side='bottom',expand=True)
root.mainloop()


# In[ ]:




