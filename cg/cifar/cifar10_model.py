# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


# ==============================================================================
# And modified by Willie.
# ==============================================================================



"""Model class for Cifar10 Dataset."""
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model_base


def get_layer_parents(adjList,lidx):
  """ Returns parent layer indices for a given layer index. """
  return [e[0] for e in adjList if e[1]==lidx]


class ConvNetCifar10(model_base.ResNet):
  """Cifar10 model with ResNetV1 and basic residual block."""

  def __init__(self,
               nnObj,
               is_training,
               batch_norm_decay,
               batch_norm_epsilon,
               data_format='channels_first'):
    super(ConvNetCifar10, self).__init__(
        is_training,
        data_format,
        batch_norm_decay,
        batch_norm_epsilon
    )
    self.num_classes = 10 + 1
    self.nnObj = nnObj


  def _get_layers(self,nn,lidx):
    # Inputs:
    #   nn - neural_network object
    #   lidx - layer index
    layerStr = nn.layer_labels[lidx]
    strideVal = nn.strides[lidx]
    poolSizeVal = 2
    stridePoolVal = 2
    num_filters = nn.num_units_in_each_layer[lidx]
    if num_filters==None: num_filters=1
    if layerStr=='softmax': num_filters=self.num_classes
    if layerStr=='relu':
      return lambda x: self._relu(x)
    elif layerStr=='conv3':
      return lambda x: self._myConv(x,3,num_filters,strideVal)
    elif layerStr=='conv5':
      return lambda x: self._myConv(x,5,num_filters,strideVal)
    elif layerStr=='conv7':
      return lambda x: self._myConv(x,7,num_filters,strideVal)
    elif layerStr=='conv9':
      return lambda x: self._myConv(x,9,num_filters,strideVal)
    elif layerStr=='res3':
      return lambda x: self._residual_v1(x,3,num_filters,num_filters,strideVal)
    elif layerStr=='res5':
      return lambda x: self._residual_v1(x,5,num_filters,num_filters,strideVal)
    elif layerStr=='res7':
      return lambda x: self._residual_v1(x,7,num_filters,num_filters,strideVal)
    elif layerStr=='res9':
      return lambda x: self._residual_v1(x,9,num_filters,num_filters,strideVal)
    elif layerStr=='avg-pool':
      return lambda x: self._avg_pool(x,poolSizeVal,stridePoolVal)
    elif layerStr=='max-pool':
      return lambda x: self._max_pool(x,poolSizeVal,stridePoolVal)
    elif layerStr=='fc':
      return lambda x: self._fully_connected(x,num_filters)
    elif layerStr=='softmax':
      return lambda x: self._softmax(x,num_filters)


  def forward_pass(self, x, input_data_format='channels_last'):
    """Build the core model within the graph."""
    if self._data_format != input_data_format:
      if input_data_format == 'channels_last':
        # Computation requires channels_first.
        x = tf.transpose(x, [0, 3, 1, 2])
      else:
        # Computation requires channels_last.
        x = tf.transpose(x, [0, 2, 3, 1])

    # Image standardization.
    x = x / 128 - 1
    nn = self.nnObj

    # Loop over layers and define conv net 
    layers = [x]  # Add first layer to layers-list 
    for lidx in range(1,nn.num_internal_layers+1):
      # Find and concatenate parent layers
      plist = get_layer_parents(nn.conn_mat.viewkeys(),lidx)
      parent_layers = [layers[i] for i in plist]
      input_layer = tf.concat(parent_layers,1)
      # Add layer to layers-list
      nextLayer = self._get_layers(nn,lidx)
      layers.append(nextLayer(input_layer))

    # Define output layer
    plist = get_layer_parents(nn.conn_mat.viewkeys(),lidx+1)
    parent_layers = [layers[i] for i in plist]
    return parent_layers


  def forward_pass_resnet(self, x, input_data_format='channels_last'):
    """Build the core model within the graph."""
    if self._data_format != input_data_format:
      if input_data_format == 'channels_last':
        # Computation requires channels_first.
        x = tf.transpose(x, [0, 3, 1, 2])
      else:
        # Computation requires channels_last.
        x = tf.transpose(x, [0, 2, 3, 1])

    # Filter and stride parameters
    theFilters = [16, 16, 32, 64]
    theStrides = [1, 2, 2]

    # Image standardization.
    x = x / 128 - 1

    x = self._conv(x, 3, 16, 1)
    x = self._batch_norm(x)
    x = self._relu(x)

    # Use basic (non-bottleneck) block and ResNet V1 (post-activation).
    res_func = self._residual_v1

    # 3 stages of block stacking.
    for i in range(3):
      with tf.name_scope('stage'):
        for j in range(7):
          if j == 0:
            # First block in a stage, filters and strides may change.
            x = res_func(x, 3, theFilters[i], theFilters[i + 1],
                         theStrides[i])
          else:
            # Following blocks in a stage, constant filters and unit stride.
            x = res_func(x, 3, theFilters[i + 1], theFilters[i + 1], 1)

    x = self._global_avg_pool(x)
    x = self._fully_connected(x, self.num_classes)

    return x


class ResNetCifar10(model_base.ResNet):
  """Cifar10 model with ResNetV1 and basic residual block."""

  def __init__(self,
               num_layers,
               is_training,
               batch_norm_decay,
               batch_norm_epsilon,
               data_format='channels_first'):
    super(ResNetCifar10, self).__init__(
        is_training,
        data_format,
        batch_norm_decay,
        batch_norm_epsilon
    )
    self.n = (num_layers - 2) // 6
    # Add one in case label starts with 1. No impact if label starts with 0.
    self.num_classes = 10 + 1
    self.filters = [16, 16, 32, 64]
    self.strides = [1, 2, 2]

  def forward_pass(self, x, input_data_format='channels_last'):
    """Build the core model within the graph."""
    if self._data_format != input_data_format:
      if input_data_format == 'channels_last':
        # Computation requires channels_first.
        x = tf.transpose(x, [0, 3, 1, 2])
      else:
        # Computation requires channels_last.
        x = tf.transpose(x, [0, 2, 3, 1])

    # Image standardization.
    x = x / 128 - 1

    x = self._conv(x, 3, 16, 1)
    x = self._batch_norm(x)
    x = self._relu(x)

    # Use basic (non-bottleneck) block and ResNet V1 (post-activation).
    res_func = self._residual_v1

    # 3 stages of block stacking.
    for i in range(3):
      with tf.name_scope('stage'):
        for j in range(self.n):
          if j == 0:
            # First block in a stage, filters and strides may change.
            x = res_func(x, 3, self.filters[i], self.filters[i + 1],
                         self.strides[i])
          else:
            # Following blocks in a stage, constant filters and unit stride.
            x = res_func(x, 3, self.filters[i + 1], self.filters[i + 1], 1)

    x = self._global_avg_pool(x)
    x = self._fully_connected(x, self.num_classes)

    return x
