import my_model
import tensorflow as tf
import numpy as np
import keras as keras
import tkinter as tk
from tkinter import filedialog
from PIL import Image

newModel = keras.models.load_model("my_model")

newModel.compile(loss='mse',optimizer='Adam')
path = filedialog.askopenfilename(filetypes = [("JPEG files",  "*.jpg")])
pic = Image.open(path)
print(path)

def onSubmit():

    img = tf.keras.utils.load_img(
        path, target_size=(160, 160)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = newModel.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

onSubmit()