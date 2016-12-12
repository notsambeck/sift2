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
from dataset import imY2R, idct
from nolearn.lasagne import NeuralNet
import lasagne
import pickle
from PIL import Image
from lasagne import layers
from lasagne.updates import nesterov_momentum
from dataset import nextTransformNarrow, quantization
import os
import datetime

# all important increment; this is locked in for training experiments at 201
increment = 201

omega = 500    # number of images to analyze in CIFAR
imageSize = 128  # number of 'pixels' in generated images
scale = 4      # number of screen pixels for big, small
# scale is number of screen pixels per SIFT pixel


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

savednet.load_params_from('net.net')


class SiftWidget(Widget):
    counter = np.array(range(1, 3072000, 1000),
                       dtype='float32').reshape((3, 32, 32), order='F')
    images_found = NumericProperty(0)
    images_shown = NumericProperty(0)
    update_best = False
    bestImage = np.zeros((3, imageSize, imageSize))
    currentImage = np.zeros((3, imageSize, imageSize))

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
        image = imY2R(idct128(t))
        toNet = np.zeros((1, 3, 32, 32), dtype='float32')
        toNet[0] = np.divide(np.subtract(imY2R(idct(t)), 128.), 128.)
        p = savednet.predict(toNet)[0]
        if p >= .5:
            # prob = str(p)[2:4]
            self.images_found += 1
            print('Image found, probabilty:', prob, '%.   #',
                  self.images_found, 'of', self.images_shown)
            s = Image.fromarray(dataset.orderPIL(image))
            s.save(''.join([self.directory, '/', str(self.images_found),
                            '.png']))
            self.bestImage = np.divide(image, 255.)
        self.counter = np.add(self.counter, increment)
        self.counter = np.mod(self.counter, quantization)
        self.showImage()
        self.showBest()

    def showImage(self):
        self.canvas.clear()
        with self.canvas:
            for j in range(imageSize):
                for i in range(imageSize):
                    pixel = np.divide(image[:, j, i], 255.)
                    Color(*pixel)
                    Rectangle(pos=(i*scale, (3*imageSize+2-j*scale)),
                              size=(scale, scale))

    def showBest(self):
        with self.canvas:
            for j in range(imageSize):
                for i in range(imageSize):
                    pixel = self.bestImage[:, j, i]
                    Color(*pixel)
                    Rectangle(pos=((i*scale + imageSize+4)*scale,
                                   (3*imageSize-1-j)*scale),
                              size=(scale, scale))

    def showAbstract(self):
        with self.canvas:
            for j in range(imageSize):
                for i in range(imageSize):
                    pixel = self.abstractsunsetImage[:, j, i]
                    Color(*pixel)
                    Rectangle(pos=(i*scale + (imageSize+4)*scale,
                                   (imageSize-1-j)*scale),
                              size=(scale, scale))
    def showSunset(self):
        with self.canvas:
            for j in range(imageSize):
                for i in range(imageSize):
                    pixel = self.sunsetImage[:, j, i]
                    Color(*pixel)
                    Rectangle(pos=(i*scale + (imageSize+4)*scale,
                                   (imageSize-1-j)*scale),
                              size=(scale, scale))

class DualWindow(GridLayout):
    def __init__(self, **kwargs):
        super(DualWindow, self).__init__(**kwargs)
        self.cols = 2
        self.cifar = CifarWidget()
        self.add_widget(self.cifar)
        self.sift = SiftWidget()
        self.add_widget(self.sift)


class DualApp(App):
    def build(self):
        dual = DualWindow()
        Clock.schedule_interval(dual.cifar.update, 1)
        Clock.schedule_interval(dual.sift.update, .2)
        return dual


class SiftApp(App):

    def build(self):
        sift = SiftWidget()
        Clock.schedule_interval(sift.update, .033)
        return sift
