import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pandas as pd
from tqdm import tqdm

from tensorflow import keras
from keras import layers
from keras.models import Sequential
# import joblib
#
# joblib.dump(plt.clf, "clf.pkl")

tf.autograph.set_verbosity(
    level=0, alsologtostdout=False
)

import pathlib
#/content/drive/MyDrive/Mushrooms
dataset_url = "C:/Users/Me/Documents/archive/allbrains"
#data_dir = tf.keras.utils.get_file('Mushrooms', origin=dataset_url, untar=True)
data_dir = pathlib.Path(dataset_url)
train_dir = pathlib.Path("C:/Users/Me/Documents/archive/testing")
test_dir = pathlib.Path("C:/Users/Me/Documents/archive/training")

#preparing data with keras
batch_size = 32
#shrinking images down for less complexity
imgHeight = 160
imgWidth = 160
# Otherwise, it yields a tuple (images, labels), where images has shape (batch_size, image_size[0], image_size[1],
# num_channels), and labels follows the format described below.
# distribution 30:70
train_set = tf.keras.utils.image_dataset_from_directory(data_dir,
                                                        validation_split = 0.3, #20% will be used for testing
                                                        subset = "training",
                                                        seed = 123,
                                                        image_size = (imgHeight, imgWidth),
                                                        batch_size = batch_size
                                                        )
#subset is used directly with validation_split and has only values of training and validation. If training = 0.2
# then it is using 1 - 0.2 = .8 of the subset(80%), if validation it is 0.2 of subset
val_set = tf.keras.utils.image_dataset_from_directory(data_dir,
                                                        validation_split = 0.3, #20% will be used for testing
                                                        subset = "validation",
                                                        seed = 123,
                                                        image_size = (imgHeight, imgWidth),
                                                        batch_size = batch_size
                                                        )
class_names = train_set.class_names #this is basically taking names from directory
print(class_names)

#folder names for this are c-sensitive and make sure to spell it right

#input pipeline/optimize w/ autotune
AUTOTUNE = tf.data.AUTOTUNE

#pre fetching data so we dont have to while doing the training
train_set = train_set.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_set = val_set.cache().prefetch(buffer_size=AUTOTUNE)

#Conv2D - type of layer used in image recognition to help identify patterns. Images do not have
#consistent pixel placement - hence harder to locate patterns/summarizes the prescence of patters in an image
#MaxPooling - CNN passes summary to maxpooling which helps clean up the patterns even more before passing to next layer, takes greatest integer(MAX)
data_augmentation = keras.Sequential(
  [
    #layers.RandomRotation(0.1),
    layers.RandomContrast(0.1),
    layers.RandomZoom(0.1),
  ]
)

numClass = len(class_names)
model = Sequential([
  data_augmentation,
#normalize bc we dont know how much the distance varies between data
  layers.Rescaling(1./255),
#args: filter(kernel is used to filter pixels and output the summation to the FILTER - Filter is 4D if kernel is 3D)
#, kernel, padding(number of 0's padding the edges of input), activation Every layer of filters is there to capture patterns.
# For example, the first layer of filters captures patterns like edges, corners, dots etc.Subsequent layers combine \
# those patterns to make bigger patterns(like combining edges to make squares, circles, etc.) Now as we move forward in the layers,
# the patterns get more complex; hence there are larger combinations of patterns to capture.That's why we increase the filter size in
# subsequent layers to capture as many combinations as possible.
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.5),
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.5),
  layers.Flatten(),
  layers.Dense(256, activation='relu'),
  #output the amount of 4 classifications
  layers.Dense(numClass)
])

#optimizer adam - https://www.geeksforgeeks.org/intuition-of-adam-optimizer/
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#epoch is number of one complete pass of the training set through the algo/ # of complete passes
epochs= 20
history = model.fit(
  train_set,
  validation_data=val_set,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

print(acc)

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save("my_model")

