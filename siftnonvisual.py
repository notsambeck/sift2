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
import requests

# dataset is a sift module that imports CIFAR and provides
# image transform functions and access to saved datasets/etc.
import dataset
import sift_keras
from sift_keras import model

import tweepy
from secret import consumerSecret, consumerKey
from secret import accessToken, accessTokenSecret

from google.cloud import vision
vision_client = vision.Client()


# SETTINGS for twitter and siftapp

siftapp_mode = True

twitter_mode = False

if twitter_mode:
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


banned = ['computer wallpaper',
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
          'grass',
          'pink',
          'yellow',
          'wood',
          'white',
          'violet',
          'lilac',
          'lavender',
          'brown',
          'magenta',
          'angle']

boring = ['#green', '#blue', '#black', '#grass',
          '#purple', '#pink', '#light', '#sky',
          '#white', '#phenomenon', '#tree', '#water',
          '#plant', '#tree', '#macrophotography',
          '#cloud', '#plantstem', '#leaf', '#skin',
          '#flora', '#photography', '#mouth',
          '#nose', '#yellow', '#lilac', '#lavender',
          '#violet', '#face', "#photography", '#petal',
          '#eye', '#face', '#blackandwhite', '#sunlight']

bonus = ['#art', '#contemporaryart', '#painting', '#notart', '#botally',
         '#abstract', '#abstractart', '#photograph', '#notaphotograph',
         '#conceptualart', '#sift', '#notapainting']


def image_generator(increment, counter):
    '''generate a batch of images to send to net'''
    to_net = np.empty((batch_size, 32, 32, 3), 'float32')
    for i in range(batch_size):
        tr = dataset.get_transform(counter)
        to_net[i] = dataset.idct(tr)  # ycc format
        counter = np.mod(np.add(counter, increment), dataset.quantization)

    to_net = np.divide(np.subtract(to_net, scale), scale)
    # print('batch stats: max={}, min={}'.format(to_net.max(), to_net.min()))
    return to_net, counter


def Sift(increment=11999, restart=False):
    '''
    non-visualized Sift program.  Runs omega images then stops, counting by
    increment. Net checks them, candidates are saved to a folder named
    found_images/ -- today's date & increment -- / ####.png
    '''

    ########################### setup ###########################  # noqa

    last = 0          # last time an image was tweeted

    if not restart:
        print('Loading saved state...')
        try:
            with open('save.file', 'rb') as f:
                counter = pickle.load(f)
                images_found = pickle.load(f)
                processed = pickle.load(f)
                tweeted = pickle.load(f)
                print('{} images found of {} processed; tweeted {}.'
                      .format(images_found, processed*batch_size, tweeted))

        except FileNotFoundError:
            print('save.file does not exist. RESTARTING')
            counter = np.zeros((32, 32, 3), dtype='float32')
            images_found = 0
            processed = 0
            tweeted = 0
    else:
        print('Warning: Restarting. Will save over existing save.file')
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


    ########################### MAIN PROGRAM ################### noqa
    # for each batch (n = batch_size) forever (ctrl-c to quit)...
    while True:
        if processed % 100 == 0:
            print('processed {} batches of {}'.format(processed, batch_size))
        tweeted = False   # current image was not tweeted
        processed += 1

        # THIS IS IMPORTANT PART HERE ! ! ! !
        # generate batch_size images
        data, counter = image_generator(increment, counter)
        ps = model.predict_on_batch(data)

        ########################### PER IMAGE ################# noqa
        for i in range(batch_size):
            if ps[i, 1] < ps[i, 0]:   # probabilty of individual image <= 1:1
                continue

            images_found += 1
            probability = ps[i, 1]
            now = time.time()
            print('Image found: no.', images_found, ' at ', now)

            # save images to .png with PIL
            image = dataset.net2pil(data[i])
            f = ''.join([str(images_found), '_', str(probability), '.png'])
            image.save(os.path.join(directory, f))

            time_since_last = now - last
            if time_since_last < 5:  # do API / file actions after delay
                print('sleep time...')
                time.sleep(5-time_since_last)

            # save expanded/resized image to file
            arr = dataset.make_arr(image)
            expanded = dataset.expand(arr)
            exp_im = dataset.make_pil(expanded, input_format='RGB',
                                      output_format='RGB')
            exp_im.resize((512, 512)).save('most_recent.png')

            ###################### UPLOAD ######################## noqa
            if not(twitter_mode or siftapp_mode):
                continue

            ############## GOOGLE LABELS  ######################## noqa
            with open(os.path.join(directory, f), 'rb') as f:
                # get google image data
                raw = ''
                descrs = []
                labels = {}

                content = f.read()

                try:
                    goog = vision_client.image(content=content)
                    raw = goog.detect_labels()
                    labels = {label.description.replace(' ', ''): label._score
                              for label in raw}

                    # convert to hashtags because, social media?
                    descrs = ['#{}'.format(key)
                              for key in labels.keys() if key not in banned]

                except:
                    print('Google api failed, not tweeting')

            ############## TWITTER  ############################# noqa
            if twitter_mode:
                # skip boring images: no label or all boring
                if descrs == [] or all([d in boring for d in descrs]):
                    # or (descrs[0] in boring and labels[0].score < .9):
                    print('boring image, not tweeting it:', ' '.join(descrs))
                    continue

                # otherwise, compose some kind of list of things to tweet
                bot = i % 100
                if bot <= 3:
                    descrs.append('@pixelsorter')
                elif bot <= 5:
                    descrs.append('@WordPadBot')
                elif bot < 10:
                    descrs.append('@poem_exe TOPIC FOR POEM')
                elif bot < 15:
                    descrs.append('@a_quilt_bot')
                elif bot == 99:
                    # spam mode
                    my_fs = api.followers()
                    u = my_fs[np.random.randint(0, len(my_fs))]
                    u_fs = api.followers(u.screen_name)
                    usr = u_fs[np.random.randint(0, len(u_fs))]
                    at = usr.screen_name
                    ds = ['@{} IS THIS YOUR IMAGE?'
                          .format(at)] + descrs
                else:
                    for _ in range(3):
                        r = np.random.randint(0, len(bonus))
                        ds.append(bonus[r])

                # make tweet, cap length
                tweet = '''IMAGE FOUND. #{} {}'''.format(str(images_found),
                                                         ' '.join(ds))
                if len(tweet) > 130:
                    tweet = tweet[:110]

                try:
                    print('tweeting:', tweet)
                    api.update_with_media('most_recent.png', tweet)
                    last = now
                    tweeted += 1
                    tweeted = True

                except:
                    tweeted = False
                    print('Tweet failed')

            ##################  sift-app  ############################# noqa
            if siftapp_mode:
                upload('most_recent.png', "sift", 0, 1, "tweet", tweeted,
                       google_raw_data=labels)
                # TODO add raw_labels json field

    ##################   SAVE   ############################### noqa
        # save progress every 500
        if processed % 500 == 0:
            with open('save.file', 'wb') as f:
                pickle.dump(counter, f)
                pickle.dump(images_found, f)
                pickle.dump(processed, f)
                pickle.dump(tweeted, f)


def upload(filepath,
           source,
           correct_label,
           sift_label,
           description,
           tweeted,
           uploaded_by=None,
           google_raw_data=None):

    '''POST request to your API with "files" key in requests data dict'''

    base_dir = os.path.expanduser(os.path.dirname(filepath))

    # hard-coded url for API upload
    # url = 'http://localhost:8000/api/'
    url = 'https://still-taiga-56301.herokuapp.com/api/'

    file_name = os.path.basename(filepath)
    with open(os.path.join(base_dir, file_name), 'rb') as f:
        print('uploading file:', base_dir, '/', file_name)
        POST_data = {'filename': file_name,
                     'source': source,
                     'correct_label': correct_label,
                     'sift_label': sift_label,
                     'description': description,
                     'tweeted': tweeted,
                     'uploaded_by': uploaded_by,
                     'google_raw_data': str(google_raw_data)}
        files = {'filename': (file_name, f), 'file': file_name}
        resp = requests.post(url, data=POST_data, files=files)
        print(resp)


if __name__ == '__main__':
    print()
    print('SIFTnonvisual loaded. Twitter={}. For visuals, run sift.py.'
          .format(twitter_mode))
    print()
    Sift()
