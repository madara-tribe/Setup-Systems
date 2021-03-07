import tensorflow as tf
import os, sys, cv2
import numpy as np
from tensorflow.keras.callbacks import *
from tensorflow.keras.utils import to_categorical
from . import cnn_model
from tensorflow.keras.datasets import cifar10


def load_label(y, num_classes):
    y = np.array(y)
    return to_categorical(y, num_classes=num_classes)


def load_dataset():
    num_classes=10
    (X, y), (X_val, y_val) = cifar10.load_data()
    X = X/255
    X_val = X_val/255
    y = load_label(y, num_classes)
    y_val = load_label(y_val, num_classes)
    return X, y, X_val, y_val




def train():
  model = cnn_model.create_model(input_shape=(32,32,3), num_cls=10)
  
  X, y, X_val, y_val = load_dataset()

  # callback
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
  callback = [reduce_lr]

  # train
  model.fit(X, y, batch_size=4, epochs=2, callbacks=callback, validation_data=(X_val, y_val), shuffle=True)
  

  
