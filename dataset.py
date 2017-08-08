# Dataset.py contains DCT and image conversion tools and utilities to
# load CIFAR dataset, analyze it, and create images.
# Also contains low-level image transform and retrieval tools.

import numpy as np
import pickle
from PIL import Image
from scipy.fftpack import dct as scidct
import tensorflow as tf


np.set_printoptions(precision=1, suppress=True, linewidth=200,
                    edgeitems=6, threshold=128)

# import_batch is a SIFT toolset for loading pickled sets of images
# useful for testing and retraining network:
# import import_batch
# from math import cos, pi


def dct(x):
    '''dct type 2 in 2D; image -> transform'''
    out = np.empty((32, 32, 3), dtype='float32')
    for ch in range(3):
        out[:, :, ch] = scidct(scidct(x[:, :, ch], type=2,
                                      norm='ortho', axis=0),
                               type=2, norm='ortho', axis=1)
    return out


def idct(x):
    '''dct type 3 in 2D; transform -> image'''
    out = np.empty((32, 32, 3), dtype='float32')
    for ch in range(3):
        out[:, :, ch] = scidct(scidct(x[:, :, ch], type=3,
                                      norm='ortho', axis=0),
                               type=3, norm='ortho', axis=1)
    return np.clip(out, 0, 255).astype('uint8')


def expand(im):
    '''expand takes a 32x32 image and expands it by DCT/iDCT
    does not change colorspace'''
    scale_x = 16
    newsize = scale_x * 32

    tr = np.empty((32, 32, 3))
    out = np.empty((newsize, newsize, 3))

    for ch in range(3):
        tr[:, :, ch] = scidct(scidct(im[ch], type=2, norm='ortho',
                                     axis=0),
                              type=2, norm='ortho', axis=1)
    tr_pad = np.zeros((newsize, newsize, 3), dtype='float32')
    tr_pad[:32, :32] = np.multiply(tr, scale_x)
    for ch in range(3):
        out[:, :, ch] = scidct(scidct(tr_pad[ch], type=3, norm='ortho',
                                      axis=0, n=newsize),
                               type=3, norm='ortho', axis=1, n=newsize)

    return np.clip(out, 0, 255)


# optional tools for testing:
# import matplotlib.pyplot as plt

# omega is the number of samples to load or store in testing/training functions
omega = 1000


def make_transforms(data='cifar'):
    '''makeTransforms produces statistics about an image dataset (eg CIFAR)
    and a 4-d array of all their transforms for analysis / plotting.
    once assembled, you can load this pickle instead of rebuilding it
    '''
    shape = (32, 32, 3)
    if data == 'cifar':
        data = np.concatenate((importCifar10(), importCifar100()), axis=0)
    print(data.shape)

    # initialize result arrays:
    cifarMaxTransform = np.multiply(np.ones(shape, dtype='float32'), -100000)
    cifarMinTransform = np.multiply(np.ones(shape, dtype='float32'), 100000)
    total = np.zeros(shape, dtype='float32')

    # format: RGB transforms stacked numberOfImages deep
    cifarTransforms = np.zeros((len(data), 32, 32, 3),
                               dtype='float32')

    # loop through CIFAR images
    for i in range(len(data)):
        rgb = get_rgb_array(i, data)
        ycc = arr_r2y(rgb)
        transform = np.ndarray(shape, dtype='float32')
        transform = dct(ycc)
        cifarMaxTransform = np.maximum(cifarMaxTransform, transform)
        cifarMinTransform = np.minimum(cifarMinTransform, transform)
        total = np.add(total, transform)
        cifarTransforms[i] = transform
        pct = i/len(data)*100
        if round(pct) == pct:
            print('{} %'.format(pct), end='\r')
    cifarMeanTransform = np.divide(total, len(data))
    with open('init_data', 'wb') as out:
        cstd = getStdDev(cifarTransforms)

        # pickle.dump(cifarTransforms, out)    # if you want to save all data
        pickle.dump(cifarMaxTransform, out)
        pickle.dump(cifarMinTransform, out)
        pickle.dump(cifarMeanTransform, out)
        pickle.dump(cstd, out)
    return cifarTransforms


# LOAD CIFAR TRANSFORMS from pickle made in makeTransforms:
# Max, Mean, Min, Distribution = loadCifarTransforms() This should be
# the only function needed in main app once the pickle is constructed.
def loadCifarTransforms(filename='init_data'):
    # cifar-100 filename including transforms: cifarTransforms.pkl
    with open(filename, 'rb') as f:
        # cifarTransforms = pickle.load(f)   # again, if you saved data
        cifarMaxTransform = pickle.load(f)
        cifarMinTransform = pickle.load(f)
        cifarMeanTransform = pickle.load(f)
        cifarStdDeviation = pickle.load(f)

    return (cifarMaxTransform, cifarMeanTransform,
            cifarMinTransform, cifarStdDeviation)


# loads pre-pickled dataset of images. cmax is matrix of maximum
# values of transform coefs, etc.  used in creating images
cmax, cmean, cmin, cstd = loadCifarTransforms()


# find std. deviation of each transform coefficient
def getStdDev(transformArray):
    out = np.zeros((32, 32, 3), dtype='float32')
    for row in range(32):
        for col in range(32):
            for ch in range(3):
                out[row, col, ch] = transformArray[:, row, col, ch].std()
    return out


# import RGB CIFAR100 batch file of 50k images
# google cifar for this and cifar10 dataset
def importCifar100():
    # cifar-100 is 1 file
    with open('data/cifar_raw_data/cifar100.pkl', 'rb') as fo:
        u = pickle._Unpickler(fo)
        u.encoding = 'Latin1'
        cifar_data = u.load()

    cifar = cifar_data['data']
    print('Imported dataset: Cifar100.')
    print('Samples: {}, shape: {}, datarange: {} to {}.'.format(len(cifar),
                                                                cifar.shape,
                                                                cifar.min(),
                                                                cifar.max()))
    return cifar


# import RGB CIFAR10 batch files of 10k images
# return 3072 x n np.array in range 0-255
# NEEDS TO BE SCALED to +/- 1
def importCifar10(howmany=5):
    # cifar 10 data in batches 1-5
    cifar10 = np.zeros((10000*howmany, 3072), dtype='uint8')
    for i in range(1, howmany+1):
        name = ''.join(['data/cifar_raw_data/data_batch_', str(i)])
        with open(name, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'Latin1'
            cifar_data = u.load()
        cifar10[(i-1)*10000:i*10000] = cifar_data['data']
    print('Imported dataset: Cifar10')
    print('Samples: {}, shape: {}, datarange: {} to {}.'.format(len(cifar10),
                                                                cifar10.shape,
                                                                cifar10.min(),
                                                                cifar10.max()))
    return cifar10


# QUANTIZATION #

def primes_sieve(limit):
    a = [True] * limit
    a[0] = a[1] = False

    for (i, isprime) in enumerate(a):
        if isprime:
            yield i
            for n in range(i*i, limit, i):     # Mark factors non-prime
                a[n] = False


def buildPrimes(start, shape=(32, 32, 3), limit=30000):
    '''builds a diagonal array of primes, with largest values
    in the upper left of each color channel. ones for all values
    beyond limit'''
    out = np.ones(shape, dtype='float32')
    primes = primes_sieve(limit)
    # go to start value
    p = next(primes)
    while p < start:
        p = next(primes)
    # assign primes to each quantization coefficienct
    for j in range(32):
        for k in range(32):
            for ch in range(3):
                try:
                    out[31-k, 31-j, ch] = next(primes)
                except:
                    return out
    return out


''' quantization is now a matrix (size of an image) of all unique prime
numbers not really a quantization matrix but does define number of
steps allowed for transforms. It is ordered like a real quantization
matrix - less steps for less significant coeffs. Start point is
arbitrary.'''
quantization = buildPrimes(1)


def load_hard():
    f = open('10k_hard_images.pkl', 'rb')
    c = pickle.load(f)
    f.close()
    return c


# used 10/15 to save full cifar 100 datatset, should be good to go.
def save_dataset(filename, x, xt, y, yt):
    out = open(filename, 'wb')
    pickle.dump(x, out)
    pickle.dump(xt, out)
    pickle.dump(y, out)
    pickle.dump(yt, out)
    out.close()


# call: x, xt, y, yt = loadDataset('
def load_dataset(filename):
    f = open(filename, 'rb')
    x = pickle.load(f)
    xt = pickle.load(f)
    y = pickle.load(f)
    yt = pickle.load(f)
    f.close()
    return x, xt, y, yt


# pull Nth image from CIFAR data and make  32, 32, 3 numpy.array 0-255 RGB
def get_rgb_array(n, data):
    arr = np.reshape(data[n], (3, 32, 32))
    return _reorder(arr)


def _reorder(arr):
    # reorder originial 0-255 (3, x, x) image to (x, x, 3) for PIL/NN
    out = np.zeros((arr.shape[1], arr.shape[2], 3),
                   dtype='uint8')
    for i in range(3):
        out[:, :, i] = arr[i]
    return out


def make_pil(arr, input_format='YCbCr', output_format='YCbCr'):
    '''take a (32, 32, 3) array in mode input_format (default YCbCr),
    create a PIL image in YCbCr'''
    im = Image.fromarray(arr, input_format)
    if input_format != output_format:
        im = im.convert(output_format)
    return im


def make_arr(img, change_format_to=False):
    '''takes any PIL image;
    returns a np.array, by default of same format ('YCbCr' or 'RGB')'''
    if change_format_to:
        img = img.convert(change_format_to)
    return np.array(np.clip(img, 0, 255))


def arr_y2r(arr):
    ''' take YCC array
    return RGB array'''
    p = make_pil(np.clip(arr, 0, 255), output_format='RGB')
    return make_arr(p)


def arr_r2y(arr):
    ''' take rgb array;
    return ycc array'''
    p = make_pil(np.clip(arr, 0, 255), input_format='RGB')
    return make_arr(p)


def show_data(dataset, i=0):
    im = np.add(np.multiply(dataset[i], 127.5), 127.5)
    make_pil(np.clip(im, 0, 255).astype('uint8')).show()


# convert from pil image to NN data +/- 1
def pil2net(im, scale=127.5):
    if im.mode == 'RGB':
        im = im.convert('YCbCr')
    arr = np.empty((1, 32, 32, 3))
    arr[0] = np.array(im)
    return np.divide(np.subtract(arr, scale), scale)


# SIFT IMAGE (Transform) GENERATOR FUNCTIONS! #

# NEW VERSION of transform ranges Aug 2017 #

# lowest, highest are arrays of allowable values for each transform coef
# i.e. all transforms will be clipped and/or generated in this range
lowest = np.subtract(cmean, cstd)
highest = np.add(cmean, cstd)
mult = np.divide(np.multiply(cstd, 2.0), quantization)


def get_transform(count):
    '''nextTransformSimple takes a count object:
    np.ndarray, (3, chop, chop), dtype='float32'

    output values range from cmin to cmax
    (i.e. the largest value of any image transform coef. for each component
    returns a transform in YCbCr colorspace
    '''
    if count.dtype != 'float32':
        raise ValueError(count.dtype)
    return np.add(lowest, np.multiply(count, mult))


# junk show #


# vec2int converts vector NN output to an integer
# either update best image, or refine it
def vec2int(vector):
    out = 0
    biggest = 0
    for i in range(vector.shape):
        if vector[i] > biggest:
            biggest = vector[i]
            out = i
    return out


def random_transform(mean=cmean, std_dev=cstd, sigma=1):
    # generates YCC transform according to normal distribution
    # between two limit matrices generated from CIFAR
    out = np.empty((32, 32, 3), dtype='float32')
    rando = np.random.normal(loc=0, scale=sigma,
                             size=(32, 32, 3))
    out = np.add(mean, np.multiply(std_dev, rando))
    return out


# make dataset
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def build_dataset(omega, lowest=lowest, highest=highest, save=False, name=''):
    '''build_dataset makes a training set in YCbCr format.
    generates omega * classes length set
    all data are scaled to +/- 1'''
    cifar = np.append(importCifar10(), importCifar100(), axis=0)
    # make_pil(get_rgb_array(0, cifar), input_format='RGB').show()
    # make_pil(get_rgb_array(50001, cifar), input_format='RGB').show()
    print('resources loaded')

    classes = 3
    data = np.empty((classes*omega, 32, 32, 3), dtype='float32')
    labels = np.empty(classes*omega, dtype='uint8')
    count = np.zeros((32, 32, 3), dtype='float32')

    for i in range(omega):
        rgb = get_rgb_array(i, cifar)
        ycc = arr_r2y(rgb)
        tr = dct(ycc)
        capped = np.clip(tr, lowest, highest)
        data[i] = idct(capped)
        labels[i] = 1

        ycc_random = random_transform()
        data[omega+i] = idct(ycc_random)
        labels[omega+i] = 0

        count = np.add(count, 42566)
        count = np.mod(count, quantization)
        data[2*omega+i] = idct(get_transform(count))
        labels[2*omega+i] = 0
        if i % (omega // 100) == 0:
            print('{} %'.format(i * 100 / omega), end='\r')

    for e in data:
        e = np.multiply(1 + np.random.randn()/10, e)

    print('Input data range: min: {}, max: {}'.format(data.min(), data.max()))

    np.clip(data, 0.0, 255.0, out=data)
    # scale to +/- 1
    data = np.subtract(data, 127.5)
    print('centered')
    data = np.divide(data, 127.5)
    print('normalized;')
    print('output data range: min: {}, max: {}'.format(data.min(), data.max()))
    print('labels: # positive examples: {}'.format(sum(labels)))

    # shuffle data
    state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(state)
    np.random.shuffle(labels)

    if save == 'pkl':
        # data broken up to x% / 100-x% to use as desired:
        split = round(.9*classes*omega)
        save_dataset(''.join([name, '.pkl']),
                     data[0:split], data[split:classes*omega],
                     labels[0:split], labels[split:classes*omega])

    chunks = 10
    chunk_size = len(data) // chunks
    if save:
        if save == 'tfrecord':
            for chunk in range(chunks):
                filename = ''.join(['data/', name, '_', str(chunk), '.tfrec'])
                print('writing tfrecord to {}'.format(filename))
                writer = tf.python_io.TFRecordWriter(filename)
            for i in range(chunk_size):
                image_raw = data[chunk*chunk_size + i].tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(32),
                    'width': _int64_feature(32),
                    'depth': _int64_feature(3),
                    'label': _int64_feature(int(labels[i])),
                    'image_raw': _bytes_feature(image_raw)}))
                writer.write(example.SerializeToString())
                writer.close()
        elif save == 'pkl':
            for chunk in range(chunks):
                filename = ''.join(['data/', name, '_', str(chunk), '.pkl'])
                print('writing tfrecord to {}'.format(filename))
                with open(filename, 'wb') as f:
                    pickle.dump(data[chunk*chunk_size:(chunk+1)*chunk_size], f)
                    pickle.dump(labels[chunk*chunk_size:(chunk+1)*chunk_size],
                                f)
        else:
            print('that is a fake save option')

    pct = 90
    split = (omega*classes*pct)//100
    return (data[0:split], data[split:classes*omega],
            labels[0:split], labels[split:classes*omega])


def combineData(x1, x2, y1, y2):
    if x1.max() != x2.max():
        raise ValueError('different dtypes in data to combine')
    x = np.zeros((x1.shape[0]+x2.shape[0], 3, 32, 32), dtype='float32')
    y = np.zeros(x1.shape[0]+x2.shape[0], dtype='uint8')
    x[:x1.shape[0]] = x1
    x[x1.shape[0]:] = x2
    y[:x1.shape[0]] = y1
    y[x1.shape[0]:] = y2
    state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(state)
    np.random.shuffle(y)
    return x, y
