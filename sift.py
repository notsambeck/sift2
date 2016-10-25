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
from nolearn.lasagne import NeuralNet
import lasagne
import pickle
from PIL import Image
from lasagne import layers
from lasagne.updates import nesterov_momentum
from dataset import nextTransformNarrow, quantization

# all important increment; this is locked in for training experiments at 201
increment = 201

omega = 500    # number of images to analyze in CIFAR
imageSize = 32  # number of 'pixels' in generated images
scaleL = 20      # number of screen pixels for big, small
scaleS = 20
# scale is number of screen pixels per SIFT pixel


(cifarMaxTransform, cifarMeanTransform, cifarMinTransform,
 cifarTransformDistribution) = dataset.loadCifarTransforms()

savednet = NeuralNet(
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
    update_learning_rate=0.1,
    update_momentum=.9,
    max_epochs=1000,
    verbose=1,
    regression=False)

savednet.load_params_from('net/deepnet_v4.nn')


class ImagePickler(pickle.Pickler):
    def persistent_id(self, obj):
        return(obj[1])


class SiftWidget(Widget):
    counter = np.array(range(1, 3072000, 1000),
                       dtype='float32').reshape((3, 32, 32), order='F')
    images_found = NumericProperty(0)
    images_shown = NumericProperty(0)
    best = 0.0
    bestImage = np.zeros((3, imageSize, imageSize))

    def update(self, dt):
        self.canvas.clear()
        self.counter = np.add(self.counter, increment)
        self.counter = np.mod(self.counter, quantization)
        self.showImage()
        self.showBest()

    def showImage(self):
        # uses dataset.genycc on loaded data from pickle
        # t = dataset.genycc(cifarMaxTransform, cifarMinTransform, sigma='.03')
        t = nextTransformNarrow(self.counter)
        self.updateBest = 0
        self.images_shown += 1
        image = dataset.imY2R(dataset.idct(t))
        toNet = np.zeros((1, 3, 32, 32), dtype='float32')
        toNet[0] = np.divide(np.subtract(image, 128.), 128.)
        p = savednet.predict(toNet)[0]
        if p >= .5:
            prob = str(p)[2:4]
            self.images_found += 1
            print('Image found, probabilty:', prob, '%.   #',
                  self.images_found, 'of', self.images_shown)
            s = Image.fromarray(dataset.orderPIL(image))
            s.save(''.join(['found161024/', str(self.images_found), '_',
                           prob, '.png']))
            self.best = max(self.best, p)
            self.bestImage = np.divide(image, 255.)

        with self.canvas:
            for j in range(imageSize):
                for i in range(imageSize):
                    pixel = np.divide(image[:, j, i], 255.)
                    Color(*pixel)
                    Rectangle(pos=(i*scaleL, (imageSize-1-j)*scaleL),
                              size=(scaleL, scaleL))

    def showBest(self):
        with self.canvas:
            for j in range(imageSize):
                for i in range(imageSize):
                    pixel = self.bestImage[:, j, i]
                    Color(*pixel)
                    Rectangle(pos=(i*scaleS + (imageSize+4)*scaleL,
                                   (imageSize-1-j)*scaleS),
                              size=(scaleS, scaleS))


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


class CifarWidget(Widget):
    tick = NumericProperty(0)

    def update(self, dt):
        self.canvas.clear()
        transform = cifarTransformDistribution[self.tick]
        im = dataset.imY2R(dataset.idct(dataset.chop(transform, .75)))
        self.showImage(np.divide(im, 255.))
        self.tick += 1

    def showImage(self, image):
        with self.canvas:
            for i in range(imageSize):
                for j in range(imageSize):
                    Color(*image[:, j, i])
                    Rectangle(pos=(i*scaleS+33*scaleL, (32-j)*scaleS),
                              size=(scaleS, scaleS))

