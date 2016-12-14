# Image reader and generator for SIFT
# as well as visualization with Kivy

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle
from kivy.clock import Clock
from kivy.properties import StringProperty
from kivy.properties import ObjectProperty
from kivy.core.text import Label as CoreLabel
import numpy as np
import dataset
from dataset import imY2R, idct
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from matplotlib import pyplot as plt
import lasagne
import pickle
from lasagne import layers
from lasagne.updates import nesterov_momentum
from dataset import nextTransformAdjustable, quantization
import os
import datetime

# all important increment; this is locked in for training experiments at 201
increment = 91

omega = 500    # number of images to analyze in CIFAR
imageSize = 32  # number of 'pixels' in generated images
scale = 19      # number of screen pixels for big, small
# scale is number of screen pixels per SIFT pixel
padX = 22
padY = 20


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
    images_found = 0
    images_processed_8 = 0
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

    restart = False
    if not restart:
        f = open('visualized.file', 'rb')
        counter = pickle.load(f)
        images_found = pickle.load(f)
        images_processed_8 = pickle.load(f)
        print('loading saved state... already processed:',
              images_processed_8*8, ' Already found:', images_found)
        f.close()
    
    # save routine (optional)
    save = True
    if save:
        directory = "".join(['found_images/',
                             str(datetime.date.today()),
                             'visual_increment-', str(increment)])
        if not os.path.exists('found_images'):
            os.makedirs('found_images')
            if not os.path.exists(directory):
                os.makedirs(directory)
                print('saving to:', directory)

    def update(self, dt):
        self.updateBest = False
        toNet = np.zeros((1, 3, 32, 32), dtype='float32')
        tenP = 0
        # run 8 images before showing best
        for i in range(8):
            t = nextTransformAdjustable(self.counter)
            toNet[0] = np.divide(np.subtract(imY2R(idct(t)), 128.), 128.)
            p = savednet.predict(toNet)[0]
            if p > tenP:
                tenP = p
                tenT = t
            self.counter = np.add(self.counter, increment)
            self.counter = np.mod(self.counter, quantization)
        self.images_processed_8 += 1
        self.image = imY2R(idct(tenT))
        self.prob = str(tenP)[2:8]
        self.currentLabel.text = self.prob
        self.currentLabel.refresh()
        self.currentTexture = self.currentLabel.texture
        if tenP >= .005:
            self.bestLabel.text = self.prob
            self.bestLabel.refresh()
            self.bestTexture = self.bestLabel.texture
            self.images_found += 1
            print('Image found, probabilty:', self.prob, '%.   #',
                  self.images_found, 'of', self.images_processed_8*8)
            if self.save:
                visualize.plot_saliency(savednet, toNet)
                sdi = ''.join([self.directory, '/',
                               str(self.images_found)])
                plt.savefig(''.join([sdi, '_sal_',
                                     self.prob[0], self.prob[2:],
                                     '.png']))
                visualize.plot_occlusion(savednet, toNet, tenP)
                plt.savefig(''.join([sdi, '_occ_',
                                     self.prob[0], self.prob[2:],
                                     '.png']))
                dataset.toPIL(self.image).save(''.join([sdi, '.png']))
            self.bestImage = np.divide(self.image, 255.)
        self.canvas.clear()
        self.showImage(tenP)
        self.showBest()
        if np.mod(self.images_processed_8, 128) == 0:
            print(self.images_processed_8*8, 'processed. Saving progress...')
            f = open('visualized.file', 'wb')
            pickle.dump(self.counter, f)
            pickle.dump(self.images_found, f)
            pickle.dump(self.images_processed_8, f)
            f.close()

    def showImage(self, p):
        with self.canvas:
            for j in range(imageSize):
                for i in range(imageSize):
                    pixel = np.divide(self.image[:, j, i], 255.)
                    Color(*pixel)
                    Rectangle(pos=(padX + i*scale,
                                   padY + (imageSize-j)*(scale-2)),
                              size=(scale, scale))
            Color((1, 1, 1))
            Rectangle(pos=(padX, imageSize*scale + padY),
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
                    pixel = self.bestImage[:, j, i]
                    Color(*pixel)
                    Rectangle(pos=(2 * padX + (i + imageSize)*scale,
                                   padY + (imageSize-j)*(scale-2)),
                              size=(scale, scale))
            Color((1, 1, 1))
            Rectangle(pos=(2*padX + imageSize*scale,
                           imageSize*scale + padY),
                      texture=self.bestTexture, size=(200, 50))


class SiftApp(App):

    def build(self):
        sift = SiftWidget()
        Clock.schedule_interval(sift.update, 0.001)
        return sift

if __name__ == "__main__":
    SiftApp().run()
