import argparse
import os, sys, cv2
import numpy as np
import tensorflow
from tensorflow.keras.callbacks import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils  import to_categorical
from tensorflow.keras.layers import *
from models.cnn_model import create_model
from tensorflow.keras.datasets import cifar10

def get_argparser():
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--batch_size', help='batch size', default=1, type=int)
    parser.add_argument('--num_ep', help='num epochs', default=10, type=int)
    parser.add_argument('--classes', help='num clsses', default=10, type=int)
    parser.add_argument('--weight_dir', help='folder path to save trained weight', default='weight_dir', type=str)
    parser.add_argument('--ep', help='num Epochs', default=20, type=int)
    return parser

def load_label(y, num_classes):
    y = np.array(y)
    return tensorflow.keras.utils.to_categorical(y, num_classes=num_classes)


def load_dataset():
    num_classes=10
    (X, y), (X_val, y_val) = cifar10.load_data()
    X = X/255
    X_val = X_val/255
    y = load_label(y, num_classes)
    y_val = load_label(y_val, num_classes)
    return X, y, X_val, y_val




def train():
  opts = get_argparser().parse_args()
  model = create_model(input_shape=(32,32,3), num_cls=opts.classes)
  
  X, y, X_val, y_val = load_dataset()

  # callback
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
  callback = [reduce_lr]

  # train
  model.fit(X, y, batch_size=opts.batch_size, epochs=opts.ep, callbacks=callback, validation_data=(X_val, y_val), shuffle=True)
  

if __name__ == '__main__':
  train()
  
