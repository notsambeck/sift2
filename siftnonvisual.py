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

twitter_mode = True
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

# do you want to train the network? load a dataset:
# import pickle
# x, xt, y, yt = dataset.loadDataset('data/full_cifar_plus_161026.pkl')


model.load_weights(sift_keras.savefile)

batch_size = 1000  # over 2000 kills desktop
scale = 127.5  # scale factor for +/- 1


bad_wd = ['computer wallpaper',
          'pattern',
          'texture',
          'font',
          'text',
          'line',
          'atmosphere',
          'close up',
          'closeup',
          'atmosphere of earth',
          'grass family',
          'black',
          'blue',
          'purple',
          'green',
          'material',
          'phenomenon',
          'grass']

boring = ['#green', '#blue', '#black', '#grass',
          '#purple', '#pink', '#light', '#sky',
          '#white', '#phenomenon', '#tree', '#water',
          '#plant', '#tree', '#macrophotography',
          '#cloud', '#plantstem', '#leaf', '#skin',
          '#flora', '#photography', '#mouth']

bonus = ['#art', '#contemporaryart', '#painting', '#notart',
         '#abstract', '#abstractart', '#contemporaryphotography',
         '#conceptualartist', '#snapchat', '#sift']


def image_generator(increment, counter):
    to_net = np.empty((batch_size, 32, 32, 3), 'float32')
    for i in range(batch_size):
        tr = dataset.get_transform(counter)
        to_net[i] = dataset.idct(tr)  # ycc format
        counter = np.mod(np.add(counter, increment), dataset.quantization)

    to_net = np.divide(np.subtract(to_net, scale), scale)
    # print('batch stats: max={}, min={}'.format(to_net.max(), to_net.min()))
    return to_net, counter


def Sift(increment=11999, restart=False):
    '''non-visualized Sift program.  Runs omega images then stops, counting by
    increment. Net checks them, candidates are saved to a folder named
    found_images/ -- today's date & increment -- / ####.png '''

    last = 0

    if not restart:
        print('Loading saved state...')
        try:
            f = open('save.file', 'rb')
            counter = pickle.load(f)
            images_found = pickle.load(f)
            processed = pickle.load(f)
            tweeted = pickle.load(f)
            print('{} images found of {} processed; tweeted {}.'
                  .format(images_found, processed*batch_size, tweeted))
            f.close()
        except FileNotFoundError:
            print('save.file does not exist. RESTARTING')
            counter = np.zeros((32, 32, 3), dtype='float32')
            images_found = 0
            processed = 0
            tweeted = 0
    else:
        print('Warning: Restarting, will save over progress')
        counter = np.zeros((32, 32, 3), dtype='float32')
        images_found = 0
        processed = 0
        tweeted = 0

    # make dir found_images
    if not os.path.exists('found_images'):
        os.makedirs('found_images')
    directory = "".join(['found_images/', str(datetime.date.today()),
                         '_increment-', str(increment)])
    if not os.path.exists(directory):
        os.makedirs(directory)
    print('saving to', directory)

    # MAIN LOOP
    # for rep in range(1):
    while True:
        if processed % 10 == 0:
            print('processed {} batches of {}'.format(processed, batch_size))
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
                s.save(os.path.join(directory, f))

                if now - last > 30:  # only tweet after > 30 seconds
                    # s.resize((512, 512)).save('twitter.png')
                    arr = dataset.make_arr(s)
                    x = dataset.expand(arr)
                    xim = dataset.make_pil(x, input_format='RGB',
                                           output_format='RGB')
                    xim.resize((512, 512)).save('twitter.png')

                    # twitter module
                    if twitter_mode:
                        with open(os.path.join(directory, f), 'rb') as tw:
                            content = tw.read()
                            try:
                                goog = vision_client.image(content=content)
                                labels = goog.detect_labels()
                                labels = [label for label in labels
                                          if label.description not in bad_wd]

                                # num = labels[0].score
                                # word = labels[0].description
                                # print(word, num)

                                ds = ['#'+label.description.replace(' ', '')
                                      for label in labels]

                            except:
                                print('Google api failed, not tweeting')
                                continue   # or tweet without this continue
                                tweet = '''#DEFINITELY #FOUND #AN #IMAGE.
#painting #notapainting #art #notart'''

                        # skip boring images
                        if all([d in boring for d in ds]) or \
                           (ds[0] in boring and labels[0].score < .98):
                            print('boring image, not tweeting it.')
                            print('_'.join(ds))
                            continue

                        # different kinds of tweets
                        bot = i % 100
                        if bot <= 5:
                            ds.append('@pixelsorter')

                        elif 5 < bot <= 10:
                            ds.append('@WordPadBot')

                        elif bot == 99:
                            # spam mode
                            my_fs = api.followers()
                            u = my_fs[np.random.randint(0, len(my_fs))]
                            u_fs = api.followers(u.screen_name)
                            usr = u_fs[np.random.randint(0, len(u_fs))]
                            at = usr.screen_name
                            ds = ['@{} IS THIS YOUR IMAGE?'
                                  .format(at)] + ds

                        else:
                            for _ in range(3):
                                r = np.random.randint(0, len(bonus))
                                ds.append(bonus[r])

                        # make tweet, cap length
                        tweet = '''IMAGE FOUND. #{}
{}'''.format(str(images_found), ' '.join(ds))
                        if len(tweet) > 130:
                            tweet = tweet[:110]

                        try:
                            print('tweeting:', tweet)
                            api.update_with_media('twitter.png', tweet)
                            last = now
                            tweeted += 1

                        except:
                            print('Tweet failed')

        # save progress
        if processed % 100 == 0:
            print('saving progress to save.file')
            f = open('save.file', 'wb')
            pickle.dump(counter, f)
            pickle.dump(images_found, f)
            pickle.dump(processed, f)
            pickle.dump(tweeted, f)
            f.close()


if __name__ == '__main__':
    print()
    print('SIFTnonvisual loaded. Twitter={}. For visuals, run sift.py.'
          .format(twitter_mode))
    print()
    Sift()
