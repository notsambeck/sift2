# Image reader and generator for SIFT
# as well as visualization with Kivy

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle
from kivy.clock import Clock
from kivy.properties import NumericProperty
from kivy.properties import StringProperty
from kivy.properties import ObjectProperty
from kivy.core.text import Label as CoreLabel
import numpy as np
import dataset
from dataset import imY2R, idct, imR2Y, dct
from nolearn.lasagne import NeuralNet
import lasagne
import pickle
from PIL import Image
from lasagne import layers
from lasagne.updates import nesterov_momentum
from dataset import nextTransformAdjustable, quantization
import os
import datetime

# all important increment; this is locked in for training experiments at 201
increment = 291

omega = 500    # number of images to analyze in CIFAR
imageSize = 32  # number of 'pixels' in generated images
scale = 27      # number of screen pixels for big, small
# scale is number of screen pixels per SIFT pixel
padX = 30
padY = 80


(cifarMaxTransform, cifarMeanTransform, cifarMinTransform,
 cifarStdDev) = dataset.loadCifarTransforms()

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
    output_nonlinearity=None,
    output_num_units=1,
    update=nesterov_momentum,
    update_learning_rate=0.007,
    update_momentum=.9,
    max_epochs=1000,
    verbose=2,
    regression=True)

savednet.load_params_from('regression.net')


class SiftWidget(Widget):
    counter = np.array(range(1, 3072000, 1000),
                       dtype='float32').reshape((3, 32, 32), order='F')
    images_found = NumericProperty(0)
    images_shown = NumericProperty(0)
    workingP = 0
    prob = StringProperty()
    update_best = False
    best = np.zeros((3, imageSize, imageSize))
    image = np.zeros((3, imageSize, imageSize))
    bestLabel = CoreLabel(text='test', font_size=20, color=(1, 1, 1, .8))
    currentLabel = CoreLabel(text='test', font_size=20, color=(1, 1, 1, .8))
    bestTexture = ObjectProperty()
    currentTexture = ObjectProperty()
    texture_size = ObjectProperty()
    toNet = np.zeros((1, 3, 32, 32), dtype='float32')
    slider = [0, 0]

    # save routine (optional)
    save = False
    if save:
        if not os.path.exists('found_images'):
            os.makedirs('found_images')
            directory = "".join(['found_images/', str(datetime.date.today()),
                                 '_increment-', str(increment)])
            if not os.path.exists(directory):
                os.makedirs(directory)
                print('saving to:', directory)

    def update(self, dt):
        t = nextTransformAdjustable(self.counter)
        self.updateBest = False
        self.images_shown += 1
        self.image = imY2R(idct(t))
        self.toNet[0] = np.divide(np.subtract(imY2R(idct(t)), 128.), 128.)
        p = savednet.predict(self.toNet)[0]

        self.prob = str(p)[2:8]
        self.currentLabel.text = self.prob
        self.currentLabel.refresh()
        self.currentTexture = self.currentLabel.texture

        if p >= .5:
            self.bestLabel.text = self.prob
            self.bestLabel.refresh()
            self.bestTexture = self.bestLabel.texture

            self.images_found += 1
            print('Image found, probabilty:', self.prob, '%.   #',
                  self.images_found, 'of', self.images_shown)
            self.best = np.divide(self.image, 255.)

            # save optional
            if self.save:
                s = Image.fromarray(dataset.orderPIL(self.image))
                s.save(''.join([self.directory, '/', str(self.images_found),
                                '.png']))

        else:
            self.best, self.workingP = (imageTweaker(self.best,
                                                     self.workingP,
                                                     self.slider))
            # move slider
            self.slider = np.random.randint(32, size=2)

        self.counter = np.add(self.counter, increment)
        self.counter = np.mod(self.counter, quantization)
        self.canvas.clear()
        self.showImage(p)
        self.showBest()

    def showImage(self, p):
        with self.canvas:
            for j in range(imageSize):
                for i in range(imageSize):
                    pixel = np.divide(self.image[:, j, i], 255.)
                    Color(*pixel)
                    Rectangle(pos=(padX + i*scale,
                                   padY + (imageSize-j)*scale),
                              size=(scale, scale))
            Color((1, 1, 1))
            Rectangle(pos=(padX, imageSize*scale + 2*padY),
                      texture=self.currentTexture,
                      size=(200, 50))

            if False:  # add pointer thing
                Rectangle(pos=(padX + 5 + 1200*p,
                               imageSize*scale + 2*padY),
                          size=(10, 50))
                Rectangle(pos=(padX,
                               imageSize*scale + 2*padY),
                          size=(1200, 2))

    def showBest(self):
        with self.canvas:
            for j in range(imageSize):
                for i in range(imageSize):
                    pixel = self.best[:, j, i]
                    Color(*pixel)
                    Rectangle(pos=(2 * padX + (i + imageSize)*scale,
                                   padY + (imageSize-j)*scale),
                              size=(scale, scale))
            Color((1, 1, 1))
            Rectangle(pos=(2*padX + imageSize*scale,
                           imageSize*scale + 2*padY),
                      texture=self.bestTexture, size=(200, 50))


class SiftApp(App):

    def build(self):
        sift = SiftWidget()
        Clock.schedule_interval(sift.update, 0.001)
        return sift


# THE FOLLOWING IS THE TWEAKER

cmax, cmean, cmin, cstd = dataset.loadCifarTransforms()


# imageTweaker implements ultra-shitty stochastic gradient descent on images
def imageTweaker(image, vectorP, pos, pct=1.30):
    oldP = vectorP

    debug = False
    x = pos[0]
    y = pos[1]
    if debug:
        print('tweaking, starting score =', oldP, 'on transform coef =', pos)
    transform = dct(imR2Y(image))
    hold = np.copy(image)
    for i in range(3):
        r = np.random.randn()
        if debug:
            print('was', transform[i, x, y])
        transform[i, x, y] = np.add(transform[i, x, y],
                                    np.multiply(cstd[i, x, y],
                                                r*pct))
        transform[i] = np.clip(transform[i], cmin[i], cmax[i])
        image = imY2R(idct(transform))
        toNet = np.zeros((1, 3, 32, 32), dtype='float32')
        toNet[0] = np.divide(np.subtract(image, 128.), 128.)
        holdP = savednet.predict(toNet)[[0]]
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
            image = imY2R(idct(transform))
            newP = savednet.predict(np.divide(np.subtract(image, 128.),
                                              128.))[[0]]
            if debug:
                print('therefore tried again: p is now:', newP)
    if newP < oldP:
        if debug:
            print('made it worse, revert!')
        return(hold, oldP)
    else:
        return(imY2R(idct(transform)), newP)
