# Dataset.py contains DCT and image conversion tools and utilities to
# load CIFAR dataset, analyze it, and create images.
# Also contains low-level image transform and retrieval tools.

import numpy as np
import pickle
from PIL import Image
from scipy.fftpack import dct as scidct

# import_batch is a SIFT toolset for loading pickled sets of images
# useful for testing and retraining network:
# import import_batch
# from math import cos, pi


def idct(x):
    return scidct(x, type=3, norm='ortho')


def dct(x):
    return scidct(x, type=2, norm='ortho')


def idct_expand(x):
    return scidct(x, type=3, n=128, norm='ortho')


# optional tools for testing:
# import matplotlib.pyplot as plt

# omega is the number of samples to load or store in testing/training functions
omega = 1000


# makeTransforms produces statistics about an image dataset (eg CIFAR)
# and a 4-d array of all their transforms for analysis / plotting.
# Once assembled, you can load this pickle instead of rebuilding
def makeTransforms(dataset, numberOfImages=omega):
    # initialize result arrays:
    cifarMaxTransform = np.zeros((3, 32, 32), dtype='float32')
    cifarMinTransform = np.zeros((3, 32, 32), dtype='float32')
    cifarMinTransform = dct(getImageYCC(0, dataset))
    total = np.zeros((3, 32, 32), dtype='float32')

    # format: RGB transforms stacked numberOfImages deep
    cifarTransforms = np.zeros((numberOfImages, 3, 32, 32),
                               dtype='float32')

    # loop through CIFAR images
    for i in range(numberOfImages):
        transform = dct(getImageYCC(i, dataset))
        cifarMaxTransform = np.maximum(cifarMaxTransform, transform)
        cifarMinTransform = np.minimum(cifarMinTransform, transform)
        total = np.add(total, transform)
        cifarTransforms[i] = transform
        pct = i/numberOfImages*100
        if round(pct) == pct:
            print(''.join([str(pct), '%...']))
    cifarMeanTransform = np.divide(total, numberOfImages)
    out = open('init_data', 'wb')
    cstd = getStdDev(cifarTransforms)

    # pickle.dump(cifarTransforms, out)    # if you want to save all data
    pickle.dump(cifarMaxTransform, out)
    pickle.dump(cifarMinTransform, out)
    pickle.dump(cifarMeanTransform, out)
    pickle.dump(cstd, out)
    out.close()


# LOAD CIFAR TRANSFORMS from pickle made in makeTransforms:
# Max, Mean, Min, Distribution = loadCifarTransforms() This should be
# the only function needed in main app once the pickle is constructed.
def loadCifarTransforms():
    # cifar-100 filename including transforms: cifarTransforms.pkl
    pkl_file = open('init_data', 'rb')
    # cifarTransforms = pickle.load(pkl_file)   # again, if you saved data
    cifarMaxTransform = pickle.load(pkl_file)
    cifarMinTransform = pickle.load(pkl_file)
    cifarMeanTransform = pickle.load(pkl_file)
    cifarStdDeviation = pickle.load(pkl_file)
    pkl_file.close()
    return (cifarMaxTransform, cifarMeanTransform,
            cifarMinTransform, cifarStdDeviation)


# loads pre-pickled dataset of images. cmax is matrix of maximum
# values of transform coefs, etc.  used in creating images
cmax, cmean, cmin, cstd = loadCifarTransforms()


# find std. deviation of each transform coefficient
def getStdDev(transformArray):
    out = np.zeros((3, 32, 32), dtype='float32')
    for row in range(32):
        for col in range(32):
            for chan in range(3):
                out[chan, row, col] = transformArray[:, chan, row, col].std()
    return out


# import RGB CIFAR100 batch file of 50k images
# google cifar for this and cifar10 dataset
def importCifar100():
    # cifar-100 1 file
    with open('data/cifar_raw_data/cifar100.pkl', 'rb') as fo:
        u = pickle._Unpickler(fo)
        u.encoding = 'Latin1'
        cifar_data = u.load()

    cifar100 = cifar_data['data']
    print('Imported dataset. Samples:', len(cifar100), ', shape:',
          cifar100.shape)
    return cifar100


# import RGB CIFAR10 batch files of 10k images
# return 3072 x n np.array in range 0-255
# NEEDS TO BE SCALED to +/- 1
def importCifar10(howmany=1):
    # cifar 10 data in batches 1-5
    cifar10 = np.zeros((10000*howmany, 3072), dtype='uint8')
    for i in range(1, howmany+1):
        name = ''.join(['data/cifar_raw_data/data_batch_', str(i)])
        with open(name, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'Latin1'
            cifar_data = u.load()
        cifar10[(i-1)*10000:i*10000] = cifar_data['data']
    print('Imported dataset.')
    print('Samples: {}, shape: {}, datarange: {} to {}.'.format(len(cifar10),
                                                                cifar10.shape,
                                                                cifar10.min(),
                                                                cifar10.max()))
    return cifar10


# chop is an ultra-shitty compression scheme that works well
def chop(trans, compression=1.0):
    for i in range(round(trans.shape[-1]*compression), trans.shape[-1]):
        trans[:, i, :] = 0
        trans[:, :, i] = 0
    return trans


# find Cartesian distance from origin for a transform
def transformDistance(dataset):
    omega = dataset.shape[0]
    distances = np.zeros(omega)
    for i in range(omega):
        distances[i] = np.power(np.sum(np.power(dataset[i], 2)), .5)
    return distances

# quantization = np.array(range(3172, 100, -1),
#                         dtype='float32').reshape((3, 32, 32), order='F')
# old placeholder version of quantization now replaced with prime


def primes_sieve(limit):
    a = [True] * limit
    a[0] = a[1] = False

    for (i, isprime) in enumerate(a):
        if isprime:
            yield i
            for n in range(i*i, limit, i):     # Mark factors non-prime
                a[n] = False


def buildPrimes(start, shape=(3, 32, 32), limit=50000):
    out = np.ones(shape, dtype='float32')
    primes = primes_sieve(limit)
    # go to start value
    p = next(primes)
    while p < start:
        p = next(primes)
    # assign primes to each quantization coefficienct
    for j in range(shape[1]):
        for k in range(shape[2]):
            for i in range(shape[0]):
                out[i, 31-k, 31-j] = next(primes)
    return out


# quantization is now a matrix (size of an image) of all unique prime
# numbers not really a quantization matrix but does define number of
# steps allowed for transforms. It is ordered like a real quantization
# matrix - less steps for less significant coeffs. Start point (500) is
# arbitrary.
quantization = buildPrimes(500)


# buildDataset makes a training set for network.
def buildDataset(omega):
    n = 3   # number of classes: cifar, small increment, big inc
    data = np.zeros((n*omega, 3, 32, 32), dtype='float32')
    label = np.zeros(n*omega, dtype='uint8')
    cifar = load_all_cifar()
    # hard = load_hard()    # if you have hard negative examples
    print('resources loaded')

    # build dataset block by block as 0-255 RGB
    for i in range(omega):
        data[i] = getImage255(i, cifar)
        label[i] = 1

    print('building generated images slowly...')
    data[omega:2*omega] = buildRGBIncremented(omega, inc=199)
    print('chunk1 built...')
    data[2*omega:3*omega] = buildRGBIncremented(omega, inc=3332)
    print('chunk2 built...')
    # data[n*omega-1978:] = hard[:1979]

    # scale to +/- 1
    data = np.subtract(data, 128.)
    print('centered')
    data = np.divide(data, 128.)
    print('normalized')

    # shuffle
    state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(state)
    np.random.shuffle(label)

    # data broken up to x% / 100-x% to use as desired:
    split = round(.95*n*omega)
    return (data[0:split], data[split:n*omega],
            label[0:split], label[split:n*omega])


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


def buildTransformsIncremented(omega, inc=44777):
    out = np.zeros((omega, 3, 32, 32), dtype='float32')
    count = np.zeros((3, 32, 32), dtype='float32')
    for i in range(omega):
        count = np.mod(np.add(count, inc), quantization)
        out[i] = nextTransformAdjustable(count)
    return out


# this was a float32 out, but I beleive that caused the neon blobs?
def buildRGBIncremented(omega, inc=44777):
    out = np.zeros((omega, 3, 32, 32), dtype='uint8')
    count = np.zeros((3, 32, 32), dtype='float32')
    for i in range(omega):
        count = np.mod(np.add(count, inc), quantization)
        out[i] = imY2R(idct(nextTransformAdjustable(count)))
    return out


def load_hard():
    f = open('data/hard_images_10k.pkl', 'rb')
    c = pickle.load(f)
    f.close()
    return c


def load_all_cifar():
    f = open('data/all_100k_cifar.pkl', 'rb')
    c = pickle.load(f)
    f.close()
    return c


# used 10/15 to save full cifar 100 datatset, should be good to go.
def saveDataset(filename, x, xt, y, yt):
    out = open(filename, 'wb')
    pickle.dump(x, out)
    pickle.dump(xt, out)
    pickle.dump(y, out)
    pickle.dump(yt, out)
    out.close()


# call: x, xt, y, yt = loadDataset('
def loadDataset(filename):
    f = open(filename, 'rb')
    x = pickle.load(f)
    xt = pickle.load(f)
    y = pickle.load(f)
    yt = pickle.load(f)
    f.close()
    return x, xt, y, yt


def genycc(transformMax, transformMin, chopPoint=32, center=.5, sigma=.01):
    # generates YCC transform according to normal distribution
    # between two limit matrices generated from CIFAR
    out = np.zeros((3, 32, 32), dtype='float32')
    rando = np.random.normal(loc=center, scale=sigma,
                             size=(3, 32, 32))
    out = np.add(transformMin,
                 np.multiply(np.subtract(transformMax, transformMin),
                             rando))
    for i in range(chopPoint, 32):
        out[:, :, i] = 0
        out[:, i, :] = 0
    return out


# pull Nth image from CIFAR data and make  32, 32, 3 numpy.array 0-255 RGB
def get_rgb_array(n, data):
    arr = np.reshape(data[n], (3, 32, 32))
    return arr


def _reorder_to_pil(image):
    # reorder an 0-255 (3, x, x) image to (x, x, 3) for PIL
    out = np.zeros((image.shape[1], image.shape[2], 3),
                   dtype='uint8')
    for i in range(3):
        out[:, :, i] = image[i]
    return out


def _reorder_from_pil(image):
    # reorder pil as nparray to 3, 32, 32 array for NN, etc.
    out = np.zeros((3, 32, 32), dtype='float32')
    for i in range(3):
        out[i] = image[:, :, i]
    return out


def make_pil_ycc(arr, input_format='YCbCr'):
    # take a (3, 32, 32) array in mode (default YCbCr),
    # create a PIL image in YCbCr
    im = Image.fromarray(_reorder_to_pil(arr), input_format)
    if input_format != 'YCbCr':
        im = im.convert('YCbCr')
    return im


def make_arr(img, change_format_to=False):
    # takes any PIL image;
    # returns a np.array, by default of same format ('YCbCr' or 'RGB')
    if change_format_to:
        img = img.convert(change_format_to)
    return _reorder_from_pil(np.array(img))


# convert from neural net data [(3,32,32) RGB +/- 1] to PIL image
# call only on NN data, not images
def net2pil(data, scale=127.5):
    return Image.fromarray(_reorder_to_pil(np.add(np.multiply(data, scale),
                                                  scale)))


# convert from pil image to NN data +/- 1
def pil2net(im, scale=127.5):
    if im.mode == 'YCbCr':
        im = im.convert('RGB')
    arr = _reorder_from_pil(np.array(im))
    return np.divide(np.subtract(arr, scale), scale)


# SIFT IMAGE (Transform) GENERATOR FUNCTIONS! #

# all of these are functional, but nextTransformAdjustable is used
# in SIFT for aesthetic reasons.


# narrowScale determines the range of transforms created below;
# larger values for narrowScale create more contrast-y images.
# 1.6 seems to be about right but this is totally subjective.
narrowScale = 1.7

# determines relative values of YCrCb components in nextTransformAdjustable:
scaler = np.array([[[1]],
                   [[1.2]],
                   [[1.2]]])

# precompute some variables to speed up math:
# clow is the mean of actual CIFAR image transforms less std. dev * narrowScale
# (cstd amd cmean are preloaded from pickle init.data)
clow = np.subtract(cmean, np.multiply(narrowScale, cstd))
# cmult is the range of transforms, scaled by number of steps i.e. quantization
cmult = np.divide(np.multiply(2.0*narrowScale, cstd), quantization)


# nextTransformAdjustable is current version
def nextTransformAdjustable(count):
    if count.dtype != 'float32':
        raise ValueError(count.dtype)
    transform = np.add(clow, np.multiply(count, cmult))
    new = np.multiply(transform, scaler)
    new[:, 0, 0] = transform[:, 0, 0]
    return new


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


if __name__ == '__main__':
    print('import cifar as c')
    c = importCifar10()
    print('x = trinv(c, i)')
    x = []
    for i in range(1):
        x.append(trinv(c, i))
