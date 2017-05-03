# lasagne/nolearn module for SiFT
# Sift can call these functions
# from here OK to use helper functions from dataset.py NOT OK to use
# functions from sift.py see blog.christianperone.com - convolutional
# neural networks blog post (2015)

# GPU info from desktop:
# Hardware Class: graphics card
# Model: "nVidia GF119 [GeForce GT 620 OEM]"
# Vendor: pci 0x10de "nVidia Corporation"
# Device: pci 0x1049 "GF119 [GeForce GT 620 OEM]"
# SubVendor: pci 0x10de "nVidia Corporation"


import numpy as np
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from PIL import Image
import os
import datetime
import pickle

# optional functions for network visualization, debug
'''
import import_batch
import matplotlib.pyplot as plt
import theano
from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
'''

# dataset is a sift module that imports CIFAR and provides
# image transform functions and access to saved datasets/etc.
import dataset

# call net.save_params_to('filename.pkl') to create & save file
net = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            ('conv2d3', layers.Conv2DLayer),
            ('conv2d4', layers.Conv2DLayer),
            ('dense1', layers.DenseLayer),
            ('dropout1', layers.DropoutLayer),
            ('dense2', layers.DenseLayer),
            ('dropout2', layers.DropoutLayer),
            ('dense3', layers.DenseLayer),
            ('dropout3', layers.DropoutLayer),
            ('output', layers.DenseLayer)],
    input_shape=(None, 3, 32, 32),
    conv2d1_num_filters=32,
    conv2d1_filter_size=(5, 5),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    maxpool1_pool_size=(2, 2),
    conv2d2_num_filters=64,
    conv2d2_filter_size=(4, 4),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
    maxpool2_pool_size=(2, 2),
    conv2d3_num_filters=128,
    conv2d3_filter_size=(2, 2),
    conv2d3_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d4_num_filters=128,
    conv2d4_filter_size=(2, 2),
    conv2d4_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d4_W=lasagne.init.GlorotUniform(),
    dense1_num_units=1024,
    dense1_nonlinearity=lasagne.nonlinearities.rectify,
    dropout1_p=0.5,
    dense2_num_units=1024,
    dense2_nonlinearity=lasagne.nonlinearities.rectify,
    dropout2_p=0.5,
    dense3_num_units=512,
    dense3_nonlinearity=lasagne.nonlinearities.rectify,
    dropout3_p=0.5,
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=3,
    update=nesterov_momentum,
    update_learning_rate=0.03,
    update_momentum=.9,
    max_epochs=1000,
    verbose=2,
    regression=False)

# for automated load of net CURRENT WORKING NET abstract_V2.net
net.load_params_from('abstract_v2.net')


# do you want to train the network more? build a dataset & load here:
# import pickle
# x, xt, y, yt = dataset.loadDataset('data/full_cifar_plus_161026.pkl')


# how many images to store as array in RAM
howManyToSave = 100000
# images are saved to .png by default, but also here
found_images = np.zeros((howManyToSave, 3, 32, 32), 'uint8')


# non-visualized Sift program.  Runs omega images then stops, counting by
# increment. Net checks them, candidates are saved to a folder named
# found_images/ -- today's date & increment -- / ####.png
def Sift(increment=999, omega=10**10, classes=4, restart=False, saveAll=False):
    images_found = 0
    counter = np.zeros((3, 32, 32), dtype='float32')
    images_saved = 0
    print('Increment =', increment)

    if not restart:
        print('Loading saved state...')
        f = open('save.file', 'rb')
        counter = pickle.load(f)
        images_found = pickle.load(f)
        print('Images previously found:', images_found)
        f.close()

    # make dir found_images
    if not os.path.exists('found_images'):
        os.makedirs('found_images')
    directory = "".join(['found_images/', str(datetime.date.today()),
                         '_increment-', str(increment)])
    if not os.path.exists(directory):
        os.makedirs(directory)
    print('saving to', directory)

    for i in range(1, omega):
        # save progress every 10^5th
        if not restart:
            if np.mod(i, 10**5) == 0:
                print(i, 'processed... saving progress to save.file')
                f = open('save.file', 'wb')
                pickle.dump(counter, f)
                pickle.dump(images_found, f)
                f.close()

        # create transform; turn into image and send to neural net
        t = dataset.nextTransformAdjustable(counter)
        image = dataset.imY2R(dataset.idct(t))
        toNet = np.zeros((1, 3, 32, 32), dtype='float32')
        toNet[0] = np.divide(np.subtract(image, 128.), 128.)
        p = net.predict(toNet)[0]
        counter = np.mod(np.add(counter, increment), dataset.quantization)
        if p >= 1:
            print('Class', p, 'image found:', images_found, 'of', i)
            s = Image.fromarray(dataset.orderPIL(image))
            s.save(''.join([directory, '/',
                            str(images_found),
                            '_class-', str(p),
                            '.png']))
            if images_found < howManyToSave:
                found_images[images_found] = image
            images_found += 1

        if saveAll:
            s = Image.fromarray(dataset.orderPIL(image))
            s.save(''.join([directory, '/',
                            str(images_saved),
                            '.png']))
            images_saved += 1

    print('Sifted through', omega, 'images and saved', images_found)


# check against validation set, optionally save miscategorized images
def check_accuracy(x, y, save=False):
    if y.ndim != 1:
        return 'fail - check_accuracy is for binary classification only'

    # if save=True, make directory for image files
    if save:
        if not os.path.exists('incorrect'):
            os.makedirs('incorrect')
        directory = "".join(['incorrect/', str(datetime.date.today())])
        if not os.path.exists(directory):
            os.makedirs(directory)
        print('saving miscategorized images to:', directory)

    t = ['true', 0]
    fp = ['false_pos', 0]
    fn = ['false_neg', 0]
    omega = y.shape[-1]
    p = net.predict(x)
    for i in range(omega):
        if (p[i]) == y[i]:
            t[1] += 1
        else:
            if save:
                s = dataset.toPIL(x[i])
                s.save(''.join([directory, '/', str(i),
                                '_T-', str(y[i]),
                                '_P-', str(p[i]), '.png']))
            if y[i] == 0:
                fp[1] += 1
            else:
                fn[1] += 1
    print('Of', omega, 'examples:')
    for thing in [t, fp, fn]:
        print(thing[0], ':', thing[1], '=', str(thing[1]/omega*100), '%')


# check against validation set, optionally save miscategorized images
def check_accuracy_vector(x, y, save=False):
    if y.ndim != 2:
        return 'FAIL - check_accuracy(x,y) is for binary classification only'

    # if save=True, make directory for image files
    if save:
        if not os.path.exists('incorrect'):
            os.makedirs('incorrect')
        directory = "".join(['incorrect/', str(datetime.date.today())])
        if not os.path.exists(directory):
            os.makedirs(directory)
        print('saving miscategorized images to:', directory)

    t = ['true', 0]
    fp = ['false_pos', 0]
    fn = ['false_neg', 0]
    omega = y.shape[-1]
    p = net.predict(x)
    for i in range(omega):
        if dataset.vec2int(p[i]) == dataset.vec2int(y[i]):
            t[1] += 1
        else:
            if save:
                s = dataset.toPIL(x[i])
                s.save(''.join([directory, '/', str(i),
                                '_T-', str(dataset.vec2int(y[i])), '.png']))
            if dataset.vec2int(y[i]) == 0:
                fp[1] += 1
            else:
                fn[1] += 1
    print('Of', omega, 'examples:')
    for thing in [t, fp, fn]:
        print(thing[0], ':', thing[1], '=', str(thing[1]/omega*100), '%')


if __name__ == '__main__':
    print('SIFT is running and saving images. For visuals, run file sift.py')
    print()
    Sift()