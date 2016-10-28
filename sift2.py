# Image reader and generator for SIFT
# as well as visualization with Kivy

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle
from kivy.clock import Clock
from kivy.properties import NumericProperty
from kivy.uix.gridlayout import GridLayout
import numpy as np
import dataset
from dataset import imY2R, idct, idct128
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
increment = 42552

omega = 500    # number of images to analyze in CIFAR
imageSize = 32  # number of 'pixels' in generated images
scale = 20
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
    update_best = False
    bestImage = np.zeros((3, imageSize, imageSize), dtype='float32')
    workingT = np.zeros((3, 32, 32), dtype='float32')
    workingScore = 0.0
    currentImage = np.zeros((3, imageSize, imageSize), dtype='float32')
    slider = [0, 0]
    toNet = np.zeros((1, 3, 32, 32), dtype='float32')
    pct = .15

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
        self.currentImage = imY2R(idct(t))

        self.toNet[0] = np.divide(np.subtract(imY2R(idct(t)), 128.), 128.)
        p = savednet.predict(self.toNet)[0]

        # either update best image, or refine it
        if p >= 0.5:
            prob = str(p)[2:6]
            print('')
            # save old best image (now altered)
            b = Image.fromarray(dataset.orderPIL(self.bestImage))
            b.save(''.join([self.directory, '/', str(self.images_found),
                            '_tweaked.png']))
            # now save new image
            self.images_found += 1
            print('Image found, probabilty:', prob, '%.   #',
                  self.images_found, 'of', self.images_shown)
            s = dataset.toPIL(self.toNet[0])
            s.save(''.join([self.directory, '/', str(self.images_found),
                            '.png']))
            self.bestImage = self.currentImage
            self.workingT = t
            self.pct = .15
            self.slider = [0, 0]
        else:
            self.workingT, self.workingScore = imageTweaker(self.workingT, p,
                                                            self.slider,
                                                            pct=self.pct)
            self.bestImage = imY2R(idct(self.workingT))
            # move slider
            if self.slider[0] == 0:
                self.slider[0] = self.slider[1]+1
                if self.slider[0] == 32:
                    self.pct = self.pct/2.0
                    self.slider[0] = 0
                    self.slider[1] = 0
                else:
                    self.slider[1] += 1
                    self.slider[0] -= 1

        self.counter = np.add(self.counter, increment)
        self.counter = np.mod(self.counter, quantization)

        self.canvas.clear()
        self.showBest()
        self.showImage()

    def showImage(self):
        with self.canvas:
            for j in range(imageSize):
                for i in range(imageSize):
                    pixel = np.divide(self.currentImage[:, j, i], 255.)
                    Color(*pixel)
                    Rectangle(pos=(i*scale, (imageSize-1-j)*scale),
                              size=(scale, scale))

    def showBest(self):
        with self.canvas:
            for j in range(imageSize):
                for i in range(imageSize):
                    pixel = np.divide(self.bestImage[:, j, i], 255.)
                    Color(*pixel)
                    Rectangle(pos=(i*scale + (imageSize+4)*scale,
                                   (imageSize-1-j)*scale),
                              size=(scale, scale))


class SiftApp(App):

    def build(self):
        sift = SiftWidget()
        Clock.schedule_interval(sift.update, .033)
        return sift


# imageTweaker implements ultra-shitty gradient descent on images
def imageTweaker(transform, oldP, pos, pct=.15):
    debug = False
    if debug: print('tweaking, starting score =', oldP, 'on transform coef =', pos)
    image = np.zeros((1, 3, 32, 32), dtype='float32')
    hold = np.copy(transform)
    for i in range(3):
        if debug: print('was', transform[i, pos[0], pos[1]])
        transform[i, pos[0], pos[1]] = np.add(transform[i, pos[0], pos[1]],
                                              np.multiply(cstd[i, pos[0], pos[1]],
                                                          pct))
        image[0] = imY2R(idct(transform))
        newP = savednet.predict(np.divide(np.subtract(image, 128.), 128.))
        if debug:
            print('channel =', i, ' first try newP =', newP)
            print('now:', transform[i, pos[0], pos[1]])
        if newP < oldP:
            transform[i, pos[0], pos[1]] = np.add(transform[i, pos[0], pos[1]],
                                                  np.multiply(cstd[i, pos[0], pos[1]],
                                                              (-2.)*pct))
            image[0] = imY2R(idct(transform))
            newP = savednet.predict(np.divide(np.subtract(image, 128.), 128.))
            if debug: print('therefore tried again: p is now:', newP)
    if newP < oldP:
        if debug: print('made it worse, revert!')
        return(hold, oldP)
    else:
        return(transform, newP)
