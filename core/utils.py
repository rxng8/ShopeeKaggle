

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   Author: Alex Nguyen
   Gettysburg College
"""

import sys
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import math
import os
import re

import imutils
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import cv2

import tensorflow as tf
from tensorflow.keras import layers, Model

import torch

def preprocess_img(
        src_img: np.ndarray, 
        shape,
        resize_method=tf.image.ResizeMethod.BILINEAR,
        range='tanh'
    ) -> tf.Tensor:
    # Expect image value range 0 - 255

    img = src_img
    if len(src_img.shape) == 2:
        img = tf.expand_dims(src_img, axis=-1)

    resized = tf.image.resize(
        img, 
        shape,
        method=resize_method
    )

    rescaled = None
    if range == 'tanh':
        rescaled = tf.cast(resized, dtype=float) / 255.0
        rescaled = (rescaled - 0.5) * 2 # range [-1, 1]
    elif range == 'sigmoid':
        rescaled = tf.cast(resized, dtype=float) / 255.0
    elif range == None:
        rescaled = tf.cast(resized, dtype=float)
    else:
        print("Wrong type!")
        sys.exit(1)

    # Convert to BGR
    bgr = rescaled[..., ::-1]
    return bgr

def deprocess_img(img):
    # Expect img range [-1, 1]
    # Do the rescale back to 0, 1 range, and convert from bgr back to rgb
    return (img / 2.0 + 0.5)[..., ::-1]

def show_img(img):
    if len(img.shape) == 3:
        plt.figure()
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    elif len(img.shape) == 2:
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show()

def load_image(path):
    return np.asarray(Image.open(path))


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def preprocess(img):
    # Expect img to have 1 channel, range from [0, 255]
    
    # Image resize
    prep = cv2.resize(img, (20, 20))

    # Image expand border
    prep = np.asarray(prep)
    prep = Image.fromarray(prep)
    prep = ImageOps.expand(prep,border=4,fill='black')
    prep = np.asarray(prep)

    # Cast image to range
    prep = tf.cast(prep, tf.float32) / 255.

    # Convert to 1 channel image
    if len(prep.shape) == 2:
        prep = tf.expand_dims(prep, axis=-1)

    return prep


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray





def linear_model(input_shape=(28,28)):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

def conv_model(input_shape=(28, 28, 1)):
    in_tensor = layers.Input(shape=input_shape)
    # tensor = tf.expand_dims(in_tensor, axis=-1)
    tensor = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(in_tensor)
    tensor = layers.MaxPool2D()(tensor)
    tensor = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(tensor)
    tensor = layers.MaxPool2D()(tensor)
    tensor = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(tensor)
    tensor = layers.Flatten()(tensor)
    tensor = layers.Dense(32, activation='relu')(tensor)
    tensor = layers.Dense(16, activation='relu')(tensor)
    out_tensor = layers.Dense(10)(tensor)
    
    model = Model(inputs=in_tensor, outputs=out_tensor)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    model.summary()
    return model

class ConvNet(Model):
    def __init__(self, batch_size=16):
        super().__init__()
        self.batch_size = batch_size
        self.conv1 = make_down_conv_sequence(16)
        self.conv2 = make_down_conv_sequence(32)
        self.conv3 = make_down_conv_sequence(64)
        self.flatten = tf.keras.layers.Flatten()
        self.linear1 = make_dense_layer(512)
        self.linear2 = make_dense_layer(64)
        self.out = make_dense_layer(10, activation=None)

        self.model = tf.keras.Sequential([
            self.conv1,
            self.conv2,
            self.conv3,
            self.flatten,
            self.linear1,
            self.linear2,
            self.out
        ])

    def call(self, batch_input):
        return self.model(batch_input)

### Utilities methods

def make_dense_layer(out_channels, activation='relu'):
    return tf.keras.layers.Dense(out_channels, activation=activation)

def make_conv_layer(out_channels, strides=1, activation='relu', padding='same'):
    layer = tf.keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=(4, 4),
        strides=strides,
        activation=activation,
        padding=padding
    )
    return layer

def make_dropout_layer(rate=0.5):
    return tf.keras.layers.Dropout(rate)

def make_max_pooling_layer():
    return tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        padding='same'
    )

def make_batch_norm_layer(**kwargs):
    return tf.keras.layers.BatchNormalization(**kwargs)

def make_down_conv_sequence(out_channels, **kwargs):
    return tf.keras.Sequential([
        make_conv_layer(out_channels, **kwargs),
        make_max_pooling_layer(),
        make_dropout_layer()
    ])
