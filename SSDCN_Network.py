# -*- coding: utf-8 -*-
"""

"""

from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import os
import scipy.io
import time
import pandas as pd
from random import shuffle
import random
import scipy.ndimage
from skimage.util import pad
import math
from band_attention import SEA


# ------------------------------------------define the network ----------------------------------------------------------------

x = tf.placeholder("float", [None, patch_size, patch_size, Band_MSI])
y = tf.placeholder("float", [None, patch_size, patch_size, Band_HSI])
is_training_samples = tf.placeholder("float", [None])
is_training = tf.placeholder(tf.bool)

NNN = 128  # kernel number
def conv_net(x):
    with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                      normalizer_fn=None):
        #tf.set_random_seed(0)

        # to extract intial feature
        Spe1 = slim.conv2d(x, NNN, 1, padding='SAME', scope='Spe1')

        # ------------------residual spectral-spatial blocks 1------------------  
        # 1. 1x1 spectral branch
        Spe2 = slim.conv2d(Spe1, NNN, 1, padding='SAME', scope='Sep2_1')
        Spe3 = slim.conv2d(Spe2, NNN, 1, padding='SAME', scope='Spe3_1',activation_fn=None)
        Spe3 = SEA(Spe3,name='Spe3')
        Spe3_residual = tf.nn.relu(tf.add(Spe3,Spe1)) 

        # ------------------residual spectral-spatial blocks 2------------------ 
        # 2. 1x1 spectral branch    
        Spe4 = slim.conv2d(Spe3_residual, NNN, 1, padding='SAME', scope='Sep4_1')
        Spe5 = slim.conv2d(Spe4, NNN, 1, padding='SAME', scope='Spe5_1',activation_fn=None)
        Spe5 = SEA(Spe5,name='Spe5')
        Spe5_residual = tf.nn.relu(tf.add(Spe5,Spe3_residual)) 

        # ------------------residual spectral-spatial blocks 3------------------ 
        # 3. 3x3 spatial branch    
        Spa6 = slim.conv2d(Spe5_residual, NNN, 3, padding='SAME', scope='Sep6_3')
        Spa7 = slim.conv2d(Spa6, NNN, 3, padding='SAME', scope='Spe7_3',activation_fn=None)
        Spa7 = SEA(Spa7,name='Spa7')
        Spa7_residual = tf.nn.relu(tf.add(Spa7,Spe5_residual))

        # ------------------residual spectral-spatial blocks 4------------------ 
        # 4. 3x3 spatial branch    
        Spa8 = slim.conv2d(Spa7_residual, NNN, 3, padding='SAME', scope='Sep8_3')
        Spa9 = slim.conv2d(Spa8, NNN, 3, padding='SAME', scope='Spe9_3',activation_fn=None)
        Spa9 = SEA(Spa9,name='Spa9')
        Spa9_residual = tf.nn.relu(tf.add(Spa9,Spa7_residual))

        # Out
        Output_HSI = slim.conv2d(Spa9_residual, Band_HSI, 1, padding='SAME', activation_fn=None)  


        #----------------------------------------Dual Network--------------------------------------------
        Output_MSI = slim.conv2d(Output_HSI, Band_MSI, 1, padding='SAME', activation_fn=None)  


    return Output_HSI, Output_MSI

# Construct model
pred, pred_MSI = conv_net(x)

    
