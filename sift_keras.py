'''
sift_keras is a module that defines the Keras neural net used by Sift,
provides functions to train the net,
and allows import by other modules.
'''

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Dropout, Dense, Activation
import numpy as np
from numpy.random import permutation, randint
import pickle
import dataset
import os
# import h5py  # h5py library is required, but import is not

# current neural net weights file for saving and loading
savefile = 'net/keras_net_v0_2017aug7.h5'


# define Keras convnet; compile model
model = Sequential()

model.add(Conv2D(32, (5, 5), padding='same', input_shape=(32, 32, 3),
                 data_format='channels_last'))
model.add(Activation('relu'))
model.add(Conv2D(64, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(units=1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(units=512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(2))
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


model.load_weights(savefile)  # load existing weights

# dataset.py and import_batch.py provide tools for creating these datasets
# using CIFAR and tiny image dataset
orig_data = 'data/keras_dataset_300k_{}.pkl'               # 0-9
import_batch_dset1 = 'data/import_batch_20170813_{}.pkl'   # 0-9
includes_tiny_images = 'data/includes_tiny_images_{}.pkl'  # 0-9

filelist1 = [import_batch_dset1.format(i) for i in range(9)]
filelist2 = [orig_data.format(i) for i in range(10)]
filelist3 = [includes_tiny_images.format(i) for i in range(10)]

testfile = 'data/import_batch_20170813_9.pkl'

filelist = filelist1 + filelist2 + filelist3


def train_net(load=savefile):
    '''train_net loads parameters from savefile by default,
    and then trains 100 epochs on multi-file dataset
    specified by filelist'''
    if load:
        try:
            model.load_weights(load)
        except:
            input('Warning: failed to load weights; will write to file {}'
                  .format(savefile))
    else:
        print('WARNING: not saving progress.')

    print()
    for epoch in range(1000):
        for chunk in permutation(filelist):
            with open(chunk, 'rb') as f:
                x = pickle.load(f)
                y = pickle.load(f)
                l = len(x)
            if l != len(y):
                raise ValueError('data/labels not equal length!')
            if len(y.shape) == 1:
                y = keras.utils.to_categorical(y, 2)

            model.fit(x, y, epochs=1, batch_size=2000)

        print('trained epoch {}; testing...'.format(epoch))
        with open(testfile, 'rb') as f:
            x = pickle.load(f)
            y = pickle.load(f)
            y = keras.utils.to_categorical(y, 2)
        e = model.evaluate(x, y, batch_size=2000)
        print(e)
        model.save(load)


def predict_incorrect(filename=testfile, limit=100, output='save'):
    '''
    load a pickle file containing images and labels;
    output up to *limit* images according to mode output
    that are miscategorized by model
    '''
    with open(filename, 'rb') as f:
        x = pickle.load(f)
        y = pickle.load(f)  # integer 0/1

    prs = model.predict(x)

    incorrect = 0
    for i in range(len(prs)):
        if prs[i, 0] > prs[i, 1]:   # predict 0
            p = 0
        else:
            p = 1

        if p != y[i]:
            incorrect += 1
            if output == 'show':
                dataset.show_data(x, i=i)
                input('prediction = {}    press enter'.format(p))
            elif output == 'save':
                im = dataset.net2pil(x[i])
                filename = ''.join([str(y[i]), 'miscat', str(i), '.png'])
                im.save(os.path.join('incorrect/keras', str(y[i]), filename))

        if incorrect == limit:
            print('limit reached at i={}'.format(i))
            return False

    print('of {} images, {} incorrect predictions'.format(len(prs),
                                                          incorrect))
    return True


# paths to various data files
tiny_images_path = "../tiny_images/tiny_images.bin"
img_count = 79302017


def load_tiny(n, show_expanded=False):
    '''load nth image from tiny_images_path'''
    with open(tiny_images_path, 'rb') as f:
        # imgs = np.empty((n, 32, 32, 3))
        f.seek(3072*n)
        arr = np.fromstring(f.read(3072), dtype='uint8')\
                .reshape((32, 32, 3), order='F')
        if show_expanded:
            dataset.make_pil(dataset.expand(arr), 'RGB').show()

        return arr


def check_real_image_detection(n=1000, chop=False):
    '''
    test precision of neural net by testing against only real images
    '''
    to_net = np.empty((n, 32, 32, 3), 'float32')
    for i in range(n):
        r = randint(0, img_count - n)
        rgb = load_tiny(r)
        ycc = dataset.arr_r2y(rgb)
        if chop:
            tr = dataset.dct(ycc)
            capped_tr = np.clip(tr, dataset.lowest, dataset.highest)
            ycc = np.clip(dataset.idct(capped_tr), 0, 255)
        to_net[i] = np.divide(np.subtract(ycc, 127.5), 127.5)

    ps = model.predict(to_net)

    wrong = 0
    for i in range(n):
        if ps[i, 0] > ps[i, 1]:
            wrong += 1
            dataset.net2pil(to_net[i]).show()

    print('of {} random tiny_images, net missed {}'.format(n, wrong))
    return wrong
