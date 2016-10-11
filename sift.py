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
from lasagne import layers
from lasagne.updates import nesterov_momentum
from siftgen import nextImageNarrow, nextImageWide, quantization

omega = 500    # number of images to analyze in CIFAR
imageSize = 32  # number of 'pixels' in generated images
scaleL = 23      # number of screen pixels for big, small
scaleS = 11
# scale is number of screen pixels per SIFT pixel


(cifarMaxTransform, cifarMeanTransform, cifarMinTransform,
 cifarTransformDistribution) = dataset.loadCifarTransforms()

savednet = NeuralNet(
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

savednet.load_params_from('regression_0927.nn')


class SiftWidget(Widget):
    counter = np.zeros((3, 32, 32))
    images_found = NumericProperty(0)
    images_shown = NumericProperty(0)
    best = 0.0
    bestImage = np.zeros((3, 32, 32))

    def update(self, dt):
        self.canvas.clear()
        self.counter = np.add(self.counter, 9777)
        self.counter = np.mod(self.counter, quantization)
        self.showImage()
        self.showBest()

    def showImage(self):
        # uses dataset.genycc on loaded data from pickle
        # t = dataset.genycc(cifarMaxTransform, cifarMinTransform, sigma='.03')
        t = nextImageNarrow(self.counter)
        self.images_shown += 1
        self.updateBest = 0
        image = dataset.imY2R(dataset.idct(dataset.chop(t, 1)))
        toNet = np.zeros((1, 3, 32, 32), dtype='float32')
        toNet[0] = np.divide(np.subtract(image, 128.), 128.)
        # p = savednet.predict(toNet)
        # turn off prediction
        if False:  # [0, 0] >= .5:
            self.images_found += 1
            print('Image found, probabilty:', p, '%.   #',
                  self.images_found, 'of', self.images_shown)
            filename = open('foundImages.pkl', 'wb')
            
            # pickle.dump(image, filename)
            filename.close()
            self.best = p[0, 0]
            self.bestImage = np.divide(image, 255.)

        with self.canvas:
            for j in range(imageSize):
                for i in range(imageSize):
                    pixel = np.divide(image[:, j, i], 255.)
                    Color(*pixel)
                    Rectangle(pos=(i*scaleL, (31-j)*scaleL),
                              size=(scaleL, scaleL))

    def showBest(self):
        with self.canvas:
            for j in range(imageSize):
                for i in range(imageSize):
                    pixel = self.bestImage[:, j, i]
                    Color(*pixel)
                    Rectangle(pos=(i*scaleS + 33*scaleL, (66-j)*scaleS),
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
