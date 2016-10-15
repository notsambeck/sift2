# lasagne nolearn module for SiFT
# Sift as master can call these functions
# from here OK to use helper functions from dataset.py NOT OK to use
# functions from sift.py see blog.christianperone.com - convolutional
# neural networks blog post 2015

import matplotlib.pyplot as plt
import pickle
import numpy as np
import theano
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# dataset is a sift module that imports CIFAR and provides
# image transform functions
import dataset

# call net.save_params_to('filename.pkl') to create & save file
net = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('dense', layers.DenseLayer),
            ('dropout2', layers.DropoutLayer),
            ('output', layers.DenseLayer)],
    input_shape=(None, 3, 32, 32),
    conv2d1_num_filters=32,
    conv2d1_filter_size=(5, 5),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.GlorotUniform(),
    maxpool1_pool_size=(2, 2),
    conv2d2_num_filters=32,
    conv2d2_filter_size=(5, 5),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
    maxpool2_pool_size=(2, 2),
    dropout1_p=0.2,
    dense_num_units=256,
    dense_nonlinearity=lasagne.nonlinearities.rectify,
    dropout2_p=0.2,
    output_nonlinearity=None,
    output_num_units=1,
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=10,
    verbose=1,
    regression=True)
