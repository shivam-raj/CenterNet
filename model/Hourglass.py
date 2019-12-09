"""Hourglass Network for Keras.
# Reference paper
- [Objects as Points]
  (https://arxiv.org/pdf/1904.07850.pdf)
# Reference implementation
- [PyTorch CenterNet]
  (https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/networks/large_hourglass.py)
- [Keras Stacked_Hourglass_Network_Keras]
  (https://github.com/yuanyuanli85/Stacked_Hourglass_Network_Keras/blob/master/src/net/hourglass.py)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from model.Residual import convolution,residual
from model.Feature import left_features,right_features,connect_left_right,bottleneck_layer,create_heads
import os

from keras.models import Model
from keras.layers import Conv2D, Input, Activation, BatchNormalization, Add, UpSampling2D, ZeroPadding2D
from keras.utils import get_file
import keras.backend as K
import numpy as np


CTDET_COCO_WEIGHTS_PATH = (
  'https://github.com/see--/keras-centernet/'
  'releases/download/0.1.0/ctdet_coco_hg.hdf5')


def normalize_image(image):
  """Normalize the image for the Hourglass network.
  # Arguments
    image: BGR uint8
  # Returns
    float32 image with the same shape as the input
  """
  mean = [0.40789655, 0.44719303, 0.47026116]
  std = [0.2886383, 0.27408165, 0.27809834]
  return ((np.float32(image) / 255.) - mean) / std


def HourglassNetwork(heads, num_stacks, cnv_dim=256, inres=(512, 512), weights='ctdet_coco',
                     dims=[256, 384, 384, 384, 512]):
  """Instantiates the Hourglass architecture.
  Optionally loads weights pre-trained on COCO.
  Note that the data format convention used by the model is
  the one specified in your Keras config at `~/.keras/keras.json`.
  # Arguments
      num_stacks: number of hourglass modules.
      cnv_dim: number of filters after the resolution is decreased.
      inres: network input shape, should be a multiple of 128.
      weights: one of `None` (random initialization),
            'ctdet_coco' (pre-training on COCO for 2D object detection),
            'hpdet_coco' (pre-training on COCO for human pose detection),
            or the path to the weights file to be loaded.
      dims: numbers of channels in the hourglass blocks.
  # Returns
      A Keras model instance.
  # Raises
      ValueError: in case of invalid argument for `weights`,
          or invalid input shape.
  """
  if not (weights in {'ctdet_coco', 'hpdet_coco', None} or os.path.exists(weights)):
    raise ValueError('The `weights` argument should be either '
                     '`None` (random initialization), `ctdet_coco` '
                     '(pre-trained on COCO), `hpdet_coco` (pre-trained on COCO) '
                     'or the path to the weights file to be loaded.')
  input_layer = Input(shape=(inres[0], inres[1], 3), name='HGInput')
  inter = pre(input_layer, cnv_dim)
  prev_inter = None
  outputs = []
  for i in range(num_stacks):
    prev_inter = inter
    _heads, inter = hourglass_module(heads, inter, cnv_dim, i, dims)
    outputs.extend(_heads)
    if i < num_stacks - 1:
      inter_ = Conv2D(cnv_dim, 1, use_bias=False, name='inter_.%d.0' % i)(prev_inter)
      inter_ = BatchNormalization(epsilon=1e-5, name='inter_.%d.1' % i)(inter_)

      cnv_ = Conv2D(cnv_dim, 1, use_bias=False, name='cnv_.%d.0' % i)(inter)
      cnv_ = BatchNormalization(epsilon=1e-5, name='cnv_.%d.1' % i)(cnv_)

      inter = Add(name='inters.%d.inters.add' % i)([inter_, cnv_])
      inter = Activation('relu', name='inters.%d.inters.relu' % i)(inter)
      inter = residual(inter, cnv_dim, 'inters.%d' % i)

  model = Model(inputs=input_layer, outputs=outputs)
  if weights == 'ctdet_coco':
    weights_path = get_file(
      '%s_hg.hdf5' % weights,
      CTDET_COCO_WEIGHTS_PATH,
      cache_subdir='models',
      file_hash='ce01e92f75b533e3ff8e396c76d55d97ff3ec27e99b1bdac1d7b0d6dcf5d90eb')
    model.load_weights(weights_path)
  elif weights == 'hpdet_coco':
    weights_path = get_file(
      '%s_hg.hdf5' % weights,
      HPDET_COCO_WEIGHTS_PATH,
      cache_subdir='models',
      file_hash='5c562ee22dc383080629dae975f269d62de3a41da6fd0c821085fbee183d555d')
    model.load_weights(weights_path)
  elif weights is not None:
    model.load_weights(weights)

  return model


def hourglass_module(heads, bottom, cnv_dim, hgid, dims):
  # create left features , f1, f2, f4, f8, f16 and f32
  lfs = left_features(bottom, hgid, dims)

  # create right features, connect with left features
  rf1 = right_features(lfs, hgid, dims)
  rf1 = convolution(rf1, 3, cnv_dim, name='cnvs.%d' % hgid)

  # add 1x1 conv with two heads, inter is sent to next stage
  # head_parts is used for intermediate supervision
  heads = create_heads(heads, rf1, hgid)
  return heads, rf1



def pre(_x, num_channels):
  # front module, input to 1/4 resolution
  _x = convolution(_x, 7, 128, name='pre.0', stride=2)
  _x = residual(_x, num_channels, name='pre.1', stride=2)
  return _x


if __name__ == '__main__':
  kwargs = {
    'num_stacks': 2,
    'cnv_dim': 256,
    'inres': (512, 512),
  }
  heads = {
    'hm': 80,
    'reg': 2,
    'wh': 2
  }
  model = HourglassNetwork(heads=heads, **kwargs)
  print(model.summary(line_length=200))
  # from IPython import embed; embed()
