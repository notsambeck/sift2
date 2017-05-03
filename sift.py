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
from dataset import imY2R, idct
from nolearn.lasagne import NeuralNet
import lasagne
import pickle
from PIL import Image
from lasagne import layers
from lasagne.updates import nesterov_momentum
from dataset import nextTransformAdjustable, quantization
import os
import datetime

# increment between transforms. Two instances of SIFT with different increments
# will see overlapping images once every LCM(i1, i2)
increment = 291

imageSize = 32  # images are 32p x 32p; DIMENSIONS ARE CURRENTLY FIXED

# screen size settings: Scale=27, padX=60, padY=80 fills 1920x1200 monitor
scale = 27      # number of screen pixels per image pixel
padX = 60       # cosmetic image padding
padY = 80

# devMode provides datasets for training
devMode = False
if devMode:
    (cifarMaxTransform, cifarMeanTransform, cifarMinTransform,
     cifarStdDev) = dataset.loadCifarTransforms()

# call net.save_params_to('filename.pkl') to create & save
# current neural net as a file
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

# choose a saved neural net to evaluate images
savednet.load_params_from('regression.net')


# Kivy widget that displays SIFT images
class SiftWidget(Widget):
    counter = np.array(range(1, 3072000, 1000),
                       dtype='float32').reshape((3, 32, 32), order='F')
    images_found = NumericProperty(0)
    images_shown = NumericProperty(0)
    prob = StringProperty()
    update_best = False
    bestImage = np.zeros((3, imageSize, imageSize))
    image = np.zeros((3, imageSize, imageSize))
    bestLabel = CoreLabel(text='test', font_size=20, color=(1, 1, 1, .8))
    currentLabel = CoreLabel(text='test', font_size=20, color=(1, 1, 1, .8))
    bestTexture = ObjectProperty()
    currentTexture = ObjectProperty()
    texture_size = ObjectProperty()
    directory = ""

    # If restart, sift will start a new image sequence from zero.
    # Otherwise it loads progress from 'visualized.file'
    restart = False
    if not restart:
        print('loading saved state...')
        f = open('visualized.file', 'rb')
        counter = pickle.load(f)
        images_found = pickle.load(f)
        f.close()

    # if save, SIFT will save a .png copy of each image it finds
    # to: --SIFT--directory--/found_images/--start--date--/####.png
    save = True
    if save:
        if not os.path.exists('found_images'):
            os.makedirs('found_images')
            directory = "".join(['found_images/',
                                 str(datetime.date.today()),
                                 'visual_increment-', str(increment)])
            if not os.path.exists(directory):
                os.makedirs(directory)
                print('saving to:', directory)

    def update(self, dt):
        t = nextTransformAdjustable(self.counter)
        self.updateBest = False
        self.images_shown += 1
        self.image = imY2R(idct(t))
        toNet = np.zeros((1, 3, 32, 32), dtype='float32')
        toNet[0] = np.divide(np.subtract(imY2R(idct(t)), 128.), 128.)
        p = savednet.predict(toNet)[0]
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
            if self.save:
                s = Image.fromarray(dataset.orderPIL(self.image))
                s.save(''.join([self.directory, '/', str(self.images_found),
                                '_viz', '.png']))
            self.bestImage = np.divide(self.image, 255.)
        self.counter = np.add(self.counter, increment)
        self.counter = np.mod(self.counter, quantization)
        self.canvas.clear()
        self.showImage(p)
        self.showBest()
        # saves progress every 10^4th image (see "restart" above)
        if np.mod(self.images_shown, 10**4) == 0:
            print(self.images_shown, 'processed... saving to visualized.file')
            f = open('visualized.file', 'wb')
            pickle.dump(self.counter, f)
            pickle.dump(self.images_found, f)
            f.close()

    # live stream of image being evaluated
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

    # most recent candidate image found by SIFT
    def showBest(self):
        with self.canvas:
            for j in range(imageSize):
                for i in range(imageSize):
                    pixel = self.bestImage[:, j, i]
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


if __name__ == "__main__":
    SiftApp().run()
