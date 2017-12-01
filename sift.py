'''
SIFT image generator and evaluator
visualized with Kivy
evaluated with Keras (tensorflow) neural net
'''

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
from dataset import idct, quantization, get_transform
from dataset import pil2net, make_arr, make_pil
import pickle
from PIL import Image
import os
import datetime

from sift_keras import model, savefile

model.load_weights(savefile)


increment = 29177

imageSize = 32  # number of 'pixels' in generated images; 32 is baked in
scale = 27      # number of screen pixels per SIFT pixel, can change

padX = 60       # center image for your screen resolution
padY = 80

# devMode provides more output and statistics for debugging/development
devMode = False
if devMode:
    (cifarMaxTransform, cifarMeanTransform, cifarMinTransform,
     cifarStdDev) = dataset.loadCifarTransforms()
    v = 2
else:
    v = 0


class SiftWidget(Widget):
    '''Kivy widget that displays SIFT images as they are generated'''
    counter = np.zeros((32, 32, 3), dtype='float32')
    images_found = NumericProperty(0)
    images_shown = NumericProperty(0)
    prob = StringProperty()
    update_best = False
    bestImage = np.zeros((imageSize, imageSize, 3))
    image = np.zeros((imageSize, imageSize, 3))
    bestLabel = CoreLabel(text='test', font_size=20, color=(1, 1, 1, .8))
    currentLabel = CoreLabel(text='test', font_size=20, color=(1, 1, 1, .8))
    bestTexture = ObjectProperty()
    currentTexture = ObjectProperty()
    texture_size = ObjectProperty()

    print("Initializing SIFT...")
    print('')

    restart = True
    if not restart:
        if os.path.exists('visualized.file'):
            f = open('visualized.file', 'rb')
            counter = pickle.load(f)
            images_found = pickle.load(f)
            print('Loading saved state...Images already found:', images_found)
            f.close()

    print('')

    # save routine (optional)
    save = False
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
        t = get_transform(self.counter)
        self.updateBest = False
        self.images_shown += 1
        im = make_pil(idct(t))
        to_net = pil2net(im)  # in ycc format
        self.image = make_arr(im, change_format_to='RGB')
        p = model.predict(to_net)
        self.prob = str(p[0, 1])
        self.currentLabel.text = self.prob
        self.currentLabel.refresh()
        self.currentTexture = self.currentLabel.texture
        # neural net section
        if p[0, 1] >= p[0, 0]:
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
        self.showImage(1)
        self.showBest()
        if np.mod(self.images_shown, 10**4) == 0:
            print(self.images_shown, 'processed... saving to visualized.file')
            f = open('visualized.file', 'wb')
            pickle.dump(self.counter, f)
            pickle.dump(self.images_found, f)
            f.close()

    def showImage(self, p):
        with self.canvas:
            for j in range(imageSize):
                for i in range(imageSize):
                    pixel = np.clip(np.divide(self.image[j, i], 255.),
                                    0, 255)
                    Color(*pixel)
                    Rectangle(pos=(padX + i*scale,
                                   padY + (imageSize-j)*scale),
                              size=(scale, scale))
            Color((1, 1, 1))
            Rectangle(pos=(padX, imageSize*scale + 2*padY),
                      texture=self.currentTexture,
                      size=(200, 50))

            if False:  # bonus graphics
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
                    pixel = self.bestImage[j, i]
                    Color(*pixel)
                    Rectangle(pos=(2 * padX + (i + imageSize)*scale,
                                   padY + (imageSize-j)*scale),
                              size=(scale, scale))
            Color((1, 1, 1))
            Rectangle(pos=(2*padX + imageSize*scale,
                           imageSize*scale + 2*padY),
                      texture=self.bestTexture, size=(200, 50))


class SiftApp(App):
    '''
    Kivy app containing Sift widget
    '''

    def build(self):
        sift = SiftWidget()
        Clock.schedule_interval(sift.update, 0.001)
        return sift


def sift():
    # for interactive use
    SiftApp().run()


if __name__ == "__main__":
    SiftApp().run()
