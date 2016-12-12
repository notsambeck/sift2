# Image reader and generator for SIFT
# as well as visualization with Kivy

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle
from kivy.clock import Clock
from kivy.properties import NumericProperty
import numpy as np
import dataset
from dataset import imY2R, idct
from dataset import nextTransformNarrow, quantization
from nolearn.lasagne import NeuralNet
import lasagne
import pickle
from PIL import Image
from lasagne import layers
from lasagne.updates import nesterov_momentum
import os
import datetime

# all important increment; this is locked in for training experiments at 201
increment = 2993

omega = 500    # number of images to analyze in CIFAR
imageSize = 32  # number of 'pixels' in generated images
scale = 16
padx = 23*scale
pady = 3*scale
# scale is number of screen pixels per SIFT pixel

cmax, cmean, cmin, cstd = dataset.loadCifarTransforms()

# call net.save_params_to('filename.pkl') to create & save file
savednet = NeuralNet(
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
    output_num_units=4,
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=.9,
    max_epochs=1000,
    verbose=2,
    regression=False)

savednet.load_params_from('abstract_v2.net')
classes = 4


class SiftWidget(Widget):
    counter = np.array(range(1, 3072000, 1000),
                       dtype='float32').reshape((3, 32, 32), order='F')
    images_found = NumericProperty(0)
    images_shown = NumericProperty(0)
    best = np.zeros((classes, 3, imageSize, imageSize), dtype='float32')
    workingT = np.zeros((3, 32, 32), dtype='float32')
    workingScore = 0.0
    currentImage = np.zeros((3, imageSize, imageSize), dtype='float32')
    slider = [0, 0]
    toNet = np.zeros((1, 3, 32, 32), dtype='float32')

    # setup save path
    if not os.path.exists('found_images'):
        os.makedirs('found_images')
    directory = "".join(['found_images/', str(datetime.date.today()),
                         '_increment-', str(increment)])
    if not os.path.exists(directory):
        os.makedirs(directory)
    print('saving to:', directory)

    def update(self, dt):
        t = nextTransformNarrow(self.counter)
        self.updateBest = False
        self.images_shown += 1
        self.best[0] = imY2R(idct(t))

        self.toNet[0] = np.divide(np.subtract(imY2R(idct(t)), 128.), 128.)
        p = savednet.predict(self.toNet)[0]

        probable = p

        if probable >= 1:
            # save old best image (now altered)
            b = Image.fromarray(dataset.orderPIL(self.best[1]))
            b.save(''.join([self.directory, '/', str(self.images_found),
                            '_alt_', str(self.workingScore)[3:7], '.png']))
            # now save new image
            self.images_found += 1
            print('Image found, probably a:', probable, ' #',
                  self.images_found, 'of', self.images_shown)
            s = dataset.toPIL(self.toNet[0])
            s.save(''.join([self.directory, '/', str(self.images_found),
                            '_', str(probable), '.png']))
            self.best[1] = self.best[0]
            self.workingT = t
            self.pct = .15
            self.workingScore = probable
            if probable == 2:
                self.best[2] = self.best[0]
            elif probable == 3:
                self.best[3] = self.best[0]
        else:
            self.best[0] = imY2R(idct(t))
            self.workingT, self.workingScore = imageTweaker(self.workingT,
                                                            self.workingScore,
                                                            self.slider)
            self.best[1] = imY2R(idct(self.workingT))
            # move slider
            self.slider = np.random.randint(32, size=2)

        self.counter = np.add(self.counter, increment)
        self.counter = np.mod(self.counter, quantization)

        self.canvas.clear()
        self.showBest()
        self.showImage()
        self.showAbstract()
        self.showSunset()

    def showImage(self):
        with self.canvas:
            for j in range(imageSize):
                for i in range(imageSize):
                    pixel = np.divide(self.best[0, :, j, i], 255.)
                    Color(*pixel)
                    Rectangle(pos=(padx+i*scale,
                                   pady+(2*imageSize+2-j)*scale),
                              size=(scale, scale))

    def showBest(self):
        with self.canvas:
            for j in range(imageSize):
                for i in range(imageSize):
                    pixel = np.divide(self.best[1, :, j, i], 255.)
                    Color(*pixel)
                    Rectangle(pos=(4*scale+padx+i*scale+imageSize*scale,
                                   pady+(2*imageSize+2-j)*scale),
                              size=(scale, scale))

    def showAbstract(self):
        with self.canvas:
            for j in range(imageSize):
                for i in range(imageSize):
                    pixel = np.divide(self.best[2, :, j, i], 255.)
                    Color(*pixel)
                    Rectangle(pos=(padx+i*scale,
                                   pady+(imageSize-1-j)*scale),
                              size=(scale, scale))

    def showSunset(self):
        with self.canvas:
            for j in range(imageSize):
                for i in range(imageSize):
                    pixel = np.divide(self.best[3, :, j, i], 255.)
                    Color(*pixel)
                    Rectangle(pos=(4*scale+padx+i*scale+imageSize*scale,
                                   pady+(imageSize-1-j)*scale),
                              size=(scale, scale))


class SiftApp(App):

    def build(self):
        sift = SiftWidget()
        Clock.schedule_interval(sift.update, .033)
        return sift


# imageTweaker implements ultra-shitty stochastic gradient descent on images
def imageTweaker(transform, vectorP, pos, pct=1.30):
    oldP = vectorP

    debug = False
    x = pos[0]
    y = pos[1]
    if debug:
        print('tweaking, starting score =', oldP, 'on transform coef =', pos)
    image = np.zeros((1, 3, 32, 32), dtype='float32')
    hold = np.copy(transform)
    for i in range(3):
        r = np.random.randn()
        if debug:
            print('was', transform[i, x, y])
        transform[i, x, y] = np.add(transform[i, x, y],
                                    np.multiply(cstd[i, x, y],
                                                r*pct))
        transform[i] = np.clip(transform[i], cmin[i], cmax[i])
        image[0] = imY2R(idct(transform))
        holdP = savednet.predict(np.divide(np.subtract(image, 128.), 128.))
        # either update best image, or refine it

        newP = holdP

        if debug:
            print('channel =', i, ' first try newP =', newP)
            print('now:', transform[i, x, y])
        if newP < oldP:
            transform[i, pos[0], pos[1]] = np.add(hold[i, x, y],
                                                  np.multiply(cstd[i, x, y],
                                                              (-1.)*pct*r))
            transform[i] = np.clip(transform[i], cmin[i], cmax[i])
            image[0] = imY2R(idct(transform))
            newP = savednet.predict(np.divide(np.subtract(image, 128.),
                                              128.))[[0]]
            if debug:
                print('therefore tried again: p is now:', newP)
    if newP < oldP:
        if debug:
            print('made it worse, revert!')
        return(hold, oldP)
    else:
        return(transform, newP)
