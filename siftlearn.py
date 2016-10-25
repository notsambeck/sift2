# lasagne nolearn module for SiFT
# Sift as master can call these functions
# from here OK to use helper functions from dataset.py NOT OK to use
# functions from sift.py see blog.christianperone.com - convolutional
# neural networks blog post 2015

# GPU speedup:
# Hardware Class: graphics card
# Model: "nVidia GF119 [GeForce GT 620 OEM]"
# Vendor: pci 0x10de "nVidia Corporation"
# Device: pci 0x1049 "GF119 [GeForce GT 620 OEM]"
# SubVendor: pci 0x10de "nVidia Corporation"


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
from PIL import Image

# dataset is a sift module that imports CIFAR and provides
# image transform functions and access to saved datasets
import dataset

# call net.save_params_to('filename.pkl') to create & save file
net = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('dropout2', layers.DropoutLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            ('conv2d3', layers.Conv2DLayer),
            ('maxpool3', layers.MaxPool2DLayer),
            ('dropout3', layers.DropoutLayer),
            ('dense1', layers.DenseLayer),
            ('dense2', layers.DenseLayer),
            ('dense3', layers.DenseLayer),
            ('output', layers.DenseLayer)],
    input_shape=(None, 3, 32, 32),
    conv2d1_num_filters=32,
    conv2d1_filter_size=(5, 5),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.GlorotUniform(),
    maxpool1_pool_size=(2, 2),
    dropout1_p=0.5,
    conv2d2_num_filters=64,
    conv2d2_filter_size=(4, 4),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
    maxpool2_pool_size=(2, 2),
    dropout2_p=0.5,
    conv2d3_num_filters=128,
    conv2d3_filter_size=(2, 2),
    conv2d3_nonlinearity=lasagne.nonlinearities.rectify,
    maxpool3_pool_size=(2, 2),
    dropout3_p=0.5,
    dense1_num_units=1024,
    dense1_nonlinearity=lasagne.nonlinearities.rectify,
    dense2_num_units=1024,
    dense2_nonlinearity=lasagne.nonlinearities.rectify,
    dense3_num_units=512,
    dense3_nonlinearity=lasagne.nonlinearities.rectify,
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=2,
    update=nesterov_momentum,
    update_learning_rate=0.03,
    update_momentum=.9,
    max_epochs=1000,
    verbose=2,
    regression=False)

#  x, xt, y, yt = dataset.loadDataset('data/cifar100_plus_narrow_50k.pkl')


def Sift(omega=100000):
    images_found = 0
    counter = np.zeros((3, 32, 32), dtype='float32')
    increment = 27711

    for i in range(omega):
        if np.mod(i, 10000) == 0:
            print(i, 'processed...')
        t = dataset.nextTransformNarrow(counter)
        image = dataset.imY2R(dataset.idct(t))
        toNet = np.zeros((1, 3, 32, 32), dtype='float32')
        toNet[0] = np.divide(np.subtract(image, 128.), 128.)
        p = net.predict(toNet)[0]
        if p == 1:
            images_found += 1
            print('Image found:', images_found, 'of', i)
            s = Image.fromarray(dataset.orderPIL(image))
            s.save(''.join(['found161024/', str(images_found), '.png']))
            counter = np.mod(np.add(counter, increment), dataset.quantization)

    print('Net searched', omega, 'images and saved', images_found)


def check_accuracy(x, y, save=False):
    t = ['true', 0]
    fp = ['false_pos', 0]
    fn = ['false_neg', 0]
    omega = y.shape[-1]
    p = net.predict(x)
    for i in range(omega):
        if p[i] == y[i]:
            t[1] += 1
        else:
            if save:
                s = dataset.toPIL(x[i])
                s.save(''.join(['wrong/', str(i), '_T', str(y[i]), '.png']))
            if y[i] == 0:
                fp[1] += 1
            else:
                fn[1] += 1
    print('Of', omega, 'examples:')
    for thing in [t, fp, fn]:
        print(thing[0], ':', thing[1], '=', str(thing[1]/omega*100), '%')


def check_accuracy_multiple(x, y, save=False):
    t = ['true', 0]
    f0 = ['false_0', 0]
    f1 = ['false_1', 0]
    f2 = ['false_2', 0]
    omega = y.shape[-1]
    p = net.predict(x)
    for i in range(omega):
        if p[i] == y[i]:
            t[1] += 1
        else:
            if save:
                s = dataset.toPIL(x[i])
                s.save(''.join(['wrong/', str(i), '_T', str(y[i]), '.png']))
            if p[i] == 0:
                f0[1] += 1
            elif p[i] == 1:
                f1[1] += 1
            else: f2[1] += 1
    print('Of', omega, 'examples:')
    for thing in [t, f0, f1, f2]:
        print(thing[0], ':', thing[1], '=', str(thing[1]/omega*100), '%')
