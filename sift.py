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
from dataset import arr_y2r, idct
import pickle
from PIL import Image
from dataset import nextTransformSimple, quantization
import os
import datetime

# all important increment
increment = 29177

imageSize = 32  # number of 'pixels' in generated images
scale = 27      # number of screen pixels per SIFT pixel

padX = 60
padY = 80

devMode = False
if devMode:
    (cifarMaxTransform, cifarMeanTransform, cifarMinTransform,
     cifarStdDev) = dataset.loadCifarTransforms()
    v = 2
else:
    v = 0


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
        t = dataset.nextTransformAdjustable(self.counter)
        self.updateBest = False
        self.images_shown += 1
        self.image = arr_y2r(idct(t))

        # neural net section
        if False:
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
        self.showImage(.75)
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
                    pixel = np.clip(np.divide(self.image[:, j, i], 255.),
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
