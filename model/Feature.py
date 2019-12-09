from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from model.Residual import convolution,residual
from keras.layers import Conv2D, Input, Activation, BatchNormalization, Add, UpSampling2D, ZeroPadding2D



def left_features(bottom, hgid, dims):
  # create left half blocks for hourglass module
  # f1, f2, f4 , f8, f16, f32 : 1, 1/2, 1/4 1/8, 1/16, 1/32 resolution
  # 5 times reduce/increase: (256, 384, 384, 384, 512)
  features = [bottom]
  for kk, nh in enumerate(dims):
    pow_str = ''
    for _ in range(kk):
      pow_str += '.center'
    _x = residual(features[-1], nh, name='kps.%d%s.down.0' % (hgid, pow_str), stride=2)
    _x = residual(_x, nh, name='kps.%d%s.down.1' % (hgid, pow_str))
    features.append(_x)
  return features


def connect_left_right(left, right, num_channels, num_channels_next, name):
  # left: 2 residual modules
  left = residual(left, num_channels_next, name=name + 'skip.0')
  left = residual(left, num_channels_next, name=name + 'skip.1')

  # up: 2 times residual & nearest neighbour
  out = residual(right, num_channels, name=name + 'out.0')
  out = residual(out, num_channels_next, name=name + 'out.1')
  out = UpSampling2D(name=name + 'out.upsampleNN')(out)
  out = Add(name=name + 'out.add')([left, out])
  return out


def bottleneck_layer(_x, num_channels, hgid):
  # 4 residual blocks with 512 channels in the middle
  pow_str = 'center.' * 5
  _x = residual(_x, num_channels, name='kps.%d.%s0' % (hgid, pow_str))
  _x = residual(_x, num_channels, name='kps.%d.%s1' % (hgid, pow_str))
  _x = residual(_x, num_channels, name='kps.%d.%s2' % (hgid, pow_str))
  _x = residual(_x, num_channels, name='kps.%d.%s3' % (hgid, pow_str))
  return _x


def right_features(leftfeatures, hgid, dims):
  rf = bottleneck_layer(leftfeatures[-1], dims[-1], hgid)
  for kk in reversed(range(len(dims))):
    pow_str = ''
    for _ in range(kk):
      pow_str += 'center.'
    rf = connect_left_right(leftfeatures[kk], rf, dims[kk], dims[max(kk - 1, 0)], name='kps.%d.%s' % (hgid, pow_str))
  return rf


def create_heads(heads, rf1, hgid):
  _heads = []
  for head in sorted(heads):
    num_channels = heads[head]
    _x = Conv2D(256, 3, use_bias=True, padding='same', name=head + '.%d.0.conv' % hgid)(rf1)
    _x = Activation('relu', name=head + '.%d.0.relu' % hgid)(_x)
    _x = Conv2D(num_channels, 1, use_bias=True, name=head + '.%d.1' % hgid)(_x)
    _heads.append(_x)
  return _heads

