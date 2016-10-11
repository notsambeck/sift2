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

# dataset will have length 2 * omega - 10,000 images in batch1
omega = 10000

# loads pre-pickled dataset of 10k cifar images
# cmax is matrix of maximum values of transform coefs, etc.
cmax, cmean, cmin, cifartransforms = dataset.loadCifarTransforms()


# dataset alternates real/fake...should be OK?
def buildDataset(channels=3):
    data = np.zeros((2*omega, channels, 32, 32), dtype='float32')
    label = np.zeros(2*omega, dtype='uint8')
    for i in range(omega):
        a = dataset.imY2R(dataset.idct(dataset.genycc(cmax, cmin)))
        # 3, 32, 32 image
        b = dataset.imY2R(dataset.idct(cifartransforms[i]))
        label[2*i] = 0
        label[2*i+1] = 1
        data[2*i] = a
        data[2*i+1] = b
    # data broken up to 75% / 25% to use as desired
    data = np.divide(np.subtract(data, 128.), 128.)
    return (data[0:1.8*omega], data[1.8*omega:2*omega],
            label[0:omega*1.8], label[1.8*omega:2*omega])


net1 = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
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
