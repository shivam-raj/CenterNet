from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from keras.models import Model
from keras.layers import Conv2D, Input, Activation, BatchNormalization, Add, UpSampling2D, ZeroPadding2D
from keras.utils import get_file
import keras.backend as K
import numpy as np

def convolution(_x, k, out_dim, name, stride=1):
  padding = (k - 1) // 2
  _x = ZeroPadding2D(padding=padding, name=name + '.pad')(_x)
  _x = Conv2D(out_dim, k, strides=stride, use_bias=False, name=name + '.conv')(_x)
  _x = BatchNormalization(epsilon=1e-5, name=name + '.bn')(_x)
  _x = Activation('relu', name=name + '.relu')(_x)
  return _x


def residual(_x, out_dim, name, stride=1):
  shortcut = _x
  num_channels = K.int_shape(shortcut)[-1]
  _x = ZeroPadding2D(padding=1, name=name + '.pad1')(_x)
  _x = Conv2D(out_dim, 3, strides=stride, use_bias=False, name=name + '.conv1')(_x)
  _x = BatchNormalization(epsilon=1e-5, name=name + '.bn1')(_x)
  _x = Activation('relu', name=name + '.relu1')(_x)

  _x = Conv2D(out_dim, 3, padding='same', use_bias=False, name=name + '.conv2')(_x)
  _x = BatchNormalization(epsilon=1e-5, name=name + '.bn2')(_x)

  if num_channels != out_dim or stride != 1:
    shortcut = Conv2D(out_dim, 1, strides=stride, use_bias=False, name=name + '.shortcut.0')(
        shortcut)
    shortcut = BatchNormalization(epsilon=1e-5, name=name + '.shortcut.1')(shortcut)

  _x = Add(name=name + '.add')([_x, shortcut])
  _x = Activation('relu', name=name + '.relu')(_x)
  return _x

