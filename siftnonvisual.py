# GPU info from desktop
# Hardware Class: graphics card
# Model: "nVidia GF119 [GeForce GT 620 OEM]"
# Vendor: pci 0x10de "nVidia Corporation"
# Device: pci 0x1049 "GF119 [GeForce GT 620 OEM]"
# SubVendor: pci 0x10de "nVidia Corporation"

import numpy as np
import os
import datetime
import time
import pickle

# dataset is a sift module that imports CIFAR and provides
# image transform functions and access to saved datasets/etc.
import dataset
import sift_keras
from sift_keras import model

twitter_mode = False
if twitter_mode:
    from google.cloud import vision
    vision_client = vision.Client()

    import tweepy
    from secret import consumerSecret, consumerKey
    from secret import accessToken, accessTokenSecret
    # secret.py is in .gitignore, stores twitter login keys as str
    auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
    auth.set_access_token(accessToken, accessTokenSecret)

    try:
        api = tweepy.API(auth)
        print('twitter connected')
        # print(api.me())
    except:
        print('twitter connect failed')
        twitter_mode = False


# optional functions for network visualization, debug
'''
import import_batch
import matplotlib.pyplot as plt
'''

# do you want to train the network more? load a dataset here:
# import pickle
# x, xt, y, yt = dataset.loadDataset('data/full_cifar_plus_161026.pkl')


model.load_weights(sift_keras.savefile)

batch_size = 1000
scale = 127.5  # scale factor for +/- 1


def image_generator(increment, counter):
    to_net = np.empty((batch_size, 32, 32, 3), 'float32')
    for i in range(batch_size):
        tr = dataset.get_transform(counter)
        to_net[i] = dataset.idct(tr)  # ycc format
        counter = np.mod(np.add(counter, increment), dataset.quantization)

    to_net = np.divide(np.subtract(to_net, scale), scale)
    # print('batch stats: max={}, min={}'.format(to_net.max(), to_net.min()))
    return to_net, counter


# non-visualized Sift program.  Runs omega images then stops, counting by
# increment. Net checks them, candidates are saved to a folder named
# found_images/ -- today's date & increment -- / ####.png
def Sift(increment=999, restart=False):

    last = time.time()

    if not restart:
        print('Loading saved state...')
        f = open('save.file', 'rb')
        counter = pickle.load(f)
        images_found = pickle.load(f)
        processed = pickle.load(f)
        print('{} images found of {} processed'.format(images_found,
                                                       processed))
        f.close()
    else:
        print('Warning: Restarting, will save over progress')
        counter = np.zeros((32, 32, 3), dtype='float32')
        images_found = 0
        processed = 0

    # make dir found_images
    if not os.path.exists('found_images'):
        os.makedirs('found_images')
    directory = "".join(['found_images/', str(datetime.date.today()),
                         '_increment-', str(increment)])
    if not os.path.exists(directory):
        os.makedirs(directory)
    print('saving to', directory)

    # for rep in range(1):
    while True:  # MAIN LOOP
        print('processed {} batches of size {}'.format(processed, batch_size))
        processed += 1
        data, counter = image_generator(increment, counter)
        ps = model.predict_on_batch(data)

        for i in range(batch_size):
            if ps[i, 1] > ps[i, 0]:
                images_found += 1
                now = time.time()
                print('Image found: no.', images_found, ' at ', now)
                # s = Image.fromarray(dataset.orderPIL(images[im]))
                s = dataset.net2pil(data[i])
                f = ''.join([str(images_found), '_', str(ps[i, 1]), '.png'])
                s.save(''.join([directory, '/', f]))

                if now - last > 3:  # only save after > 3 seconds
                    last = now
                    print('wallpaper')
                    s.resize((512, 512)).save('twitter.png')

                # tweet it
                if twitter_mode:
                    with open(''.join([directory, '/', f]), 'rb') as tw:
                        content = tw.read()
                        try:
                            bad = ['computer wallpaper',
                                   'pattern',
                                   'texture',
                                   'font',
                                   'text',
                                   'line']
                            goog = vision_client.image(content=content)
                            labels = goog.detect_labels()
                            ds = ['#'+label.description.replace(' ', '')
                                  for label in labels
                                  if label.description not in bad]
                            tweet = '''IMAGE LOCATED. #{}
{}'''.format(str(images_found), ' '.join(ds))
                            if len(tweet) > 125:
                                tweet = tweet[:105]
                        except:
                            print('Google api failed')
                            tweet = '#DEFINITE #LOCATE #IMAGE. #SIFT.'

                    try:
                        api.update_with_media('twitter.png', tweet)
                    except:
                        print('Tweet failed')

        # save progress
        if processed % 100 == 0:
            print('saving progress to save.file')
            f = open('save.file', 'wb')
            pickle.dump(counter, f)
            pickle.dump(images_found, f)
            pickle.dump(processed, f)
            f.close()


if __name__ == '__main__':
    print()
    print('SIFTnonvisual loaded. Twitter={}. For visuals, run sift.py.'
          .format(twitter_mode))
    print()
    Sift()
