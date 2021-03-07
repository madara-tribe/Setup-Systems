from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from . import scse



def create_model(input_shape=(32,32,3), num_cls=10):
  inputs = Input(shape=input_shape)
  o = Conv2D(32, (3, 3), padding='same')(inputs)
  o = channel_spatial_squeeze_excite(o)
  o = MaxPooling2D((2, 2))(o)
  o = Conv2D(64, (3, 3), padding='same')(o)
  o = scse.channel_spatial_squeeze_excite(o)
  o = MaxPooling2D((2, 2))(o)
  o = Conv2D(64, (3, 3), padding='same')(o)
  o = scse.channel_spatial_squeeze_excite(o)
  o = Dense(64, activation='relu')(o)
  o = GlobalAveragePooling2D()(o)
  o = Dense(num_cls, activation='softmax')(o)
  model = Model(inputs=inputs, outputs=o)
  model.compile(optimizer='adam', loss='categorical_crossentropy')
  return model
