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

# dataset is a sift module that imports CIFAR and provides
# image transform functions and access to saved datasets/etc.
import dataset

twitter_mode = True
if twitter_mode:
    from google.cloud import vision
    vision_client = vision.Client()

    import tweepy
    from secret import consumerSecret, consumerKey
    from secret import accessToken, accessTokenSecret
    # secret.py is in .gitignore, stores twitter login keys as str
    auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
    auth.set_access_token(accessToken, accessTokenSecret)

    try:
        api = tweepy.API(auth)
        print('twitter connected')
        # print(api.me())
    except:
        print('twitter connect failed')
        twitter_mode = False


# optional functions for network visualization, debug
'''
import import_batch
import theano
import matplotlib.pyplot as plt
from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
'''


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
net.load_params_from('170720_classification.nn')
# net.load_params_from('abstract_v2.net')

# do you want to train the network more? build a dataset & load here:
# import pickle
# x, xt, y, yt = dataset.loadDataset('data/full_cifar_plus_161026.pkl')


# how many images to store as array in RAM
howManyToSave = 100000
# saveEvery is how often to write progress
saveEvery = 10**5
batch = 1000
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

    for i in range(1, omega//batch):  # run omega/batch batches
        # save progress
        if np.mod(i, saveEvery//batch) == 0:
            print('\n', i*batch, 'processed... saving progress to save.file')
            f = open('save.file', 'wb')
            pickle.dump(counter, f)
            pickle.dump(images_found, f)
            f.close()

        if i % (saveEvery // batch // 50) == 1:
            progress = i % (saveEvery // batch)*50 // (saveEvery // batch)
            print('|{}{}| % of {}'.format('!'*progress,
                                          '_'*(50-progress),
                                          saveEvery),
                  end='\r')

        # create transforms; save batches of images as image and net obj.
        toNet = np.zeros((batch, 3, 32, 32), dtype='float32')
        images = np.zeros((batch, 3, 32, 32), dtype='float32')
        for im in range(batch):
            t = dataset.nextTransformAdjustable(counter)
            images[im] = dataset.imY2R(dataset.idct(t))
            toNet[im] = np.divide(np.subtract(images[im], 128.), 128.)
            counter = np.mod(np.add(counter, increment), dataset.quantization)

        p = net.predict(toNet)
        for im in range(batch):
            if p[im] >= 1:
                print('Class', p[im], 'image found:', images_found)
                s = Image.fromarray(dataset.orderPIL(images[im]))
                saveName = ''.join([str(images_found),
                                    '_class-',
                                    str(p[im]),
                                    '.png'])
                s.save(''.join([directory, '/', saveName]))
                images_found += 1

                # tweet it
                if twitter_mode:
                    large = s.resize((512, 512))
                    large.save('twitter.png')
                    with open(''.join([directory, '/', saveName]), 'rb') as tw:
                        content = tw.read()
                        try:
                            bad = ['computer wallpaper',
                                   'pattern',
                                   'texture',
                                   'font',
                                   'text',
                                   'line']
                            goog = vision_client.image(content=content)
                            labels = goog.detect_labels()
                            ds = ['#'+label.description.replace(' ', '')
                                  for label in labels if label.description not in bad]
                            tweet = '''IMAGE LOCATED. #{}
{}'''.format(str(images_found), ' '.join(ds))
                            if len(tweet) > 125:
                                tweet = tweet[:105]
                        except:
                            print('Google api failed')
                            tweet = '#DEFINITE #LOCATE #IMAGE. #SIFT.'

                    try:
                        api.update_with_media('twitter.png', tweet)
                    except:
                        print('Tweet failed')

            if saveAll:
                s = Image.fromarray(dataset.orderPIL(images[im]))
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
    print()
    print('SIFTnonvisual loaded. Twitter={}. For visuals, run sift.py.'.format(twitter_mode))
    print()
    Sift()
