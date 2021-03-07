import tensorflow as tf
import os, sys
import numpy as np
from tensorflow.keras.callbacks import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *



def squeeze_excite_block(input, ratio=16):
    init = input
    channel_axis = -1
    filters = input.shape[-1]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = multiply([init, se])
    return x


def spatial_squeeze_excite_block(input):
    se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False,
                kernel_initializer='he_normal')(input)

    x = multiply([input, se])
    return x


def channel_spatial_squeeze_excite(input, ratio=16):
    cse = squeeze_excite_block(input, ratio)
    sse = spatial_squeeze_excite_block(input)

    x = add([cse, sse])
    return x
    

def create_model(input_shape=(32,32,3), num_cls=10):
  inputs = Input(shape=input_shape)
  o = Conv2D(32, (3, 3), padding='same')(inputs)
  o = channel_spatial_squeeze_excite(o)
  o = MaxPooling2D((2, 2))(o)
  o = Conv2D(64, (3, 3), padding='same')(o)
  o = channel_spatial_squeeze_excite(o)
  o = MaxPooling2D((2, 2))(o)
  o = Conv2D(64, (3, 3), padding='same')(o)
  o = channel_spatial_squeeze_excite(o)
  o = Dense(64, activation='relu')(o)
  o = GlobalAveragePooling2D()(o)
  o = Dense(num_cls, activation='softmax')(o)
  model = Model(inputs=inputs, outputs=o)
  model.compile(optimizer='adam', loss='categorical_crossentropy')
  return model


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
  model = create_model(input_shape=(32,32,3), num_cls=10)
  
  X, y, X_val, y_val = load_dataset()
  # callback
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
  callback = [reduce_lr]
  # train
  model.fit(X, y, batch_size=4, epochs=2, callbacks=callback, validation_data=(X_val, y_val), shuffle=True)

if __name__=='__main__':
    train()
