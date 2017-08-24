# Dataset.py contains DCT and image conversion tools and utilities to
# load CIFAR dataset, analyze it, and create images.
# Also contains low-level image transform and retrieval tools.

import numpy as np
import pickle
from PIL import Image
from scipy.fftpack import dct as scidct


np.set_printoptions(precision=1, suppress=True, linewidth=200,
                    edgeitems=6, threshold=128)

# import_batch is a SIFT toolset for loading pickled sets of images
# useful for testing and retraining network:
# import import_batch
# from math import cos, pi


def old_dct(x):
    '''dct type 2 in 2D; image -> transform'''
    out = np.empty((32, 32, 3), dtype='float32')
    for ch in range(3):
        out[:, :, ch] = scidct(scidct(x[:, :, ch], type=2,
                                      norm='ortho', axis=0),
                               type=2, norm='ortho', axis=1)
    return out


def dct(x):
    return scidct(scidct(x, type=2, norm='ortho', axis=1),
                  type=2, norm='ortho', axis=0)


def old_idct(x):
    '''dct type 3 in 2D; transform -> image'''
    out = np.empty((32, 32, 3), dtype='float32')
    for ch in range(3):
        out[:, :, ch] = scidct(scidct(x[:, :, ch], type=3,
                                      norm='ortho', axis=0),
                               type=3, norm='ortho', axis=1)
    return np.clip(out, 0, 255).astype('uint8')


def idct(x):
    return np.clip(scidct(scidct(x, type=3, norm='ortho', axis=0),
                          type=3, norm='ortho', axis=1),
                   0, 255).astype('uint8')


def expand(im, scale_x=2):
    '''expand takes a 32x32x3 ndarray image and expands it by DCT/iDCT
    scale contrast to full range
    does not change colorspace'''
    newsize = scale_x * 32

    tr = dct(im)
    tr_pad = np.zeros((newsize, newsize, 3), dtype='float32')
    tr_pad[:32, :32, :] = np.multiply(tr, scale_x)

    lo = np.amin(tr[:, :, 0])
    # print(lo)
    hi = np.amax(tr[:, :, 0])
    # print(hi)
    tr[:, :, 0] = np.multiply(np.subtract(tr[:, :, 0], lo), 255/(hi - lo))

    out = np.empty((newsize, newsize, 3), 'float32')
    for ch in range(3):
        out[:, :, ch] = scidct(scidct(tr_pad[:, :, ch], type=3, norm='ortho',
                                      axis=0),
                               type=3, norm='ortho', axis=1)

    return np.clip(out, 0, 255).astype('uint8')


# omega is the number of samples to load or store in testing/training functions
omega = 1000


# LOAD CIFAR TRANSFORMS from pickle made in import_batch.makeTransforms:
# Max, Mean, Min, Distribution = loadCifarTransforms() This should be
# the only function needed in main app once the pickle is constructed.
def load_cifar_transforms(filename='init_data'):
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
cmax, cmean, cmin, cstd = load_cifar_transforms()


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


def get_rgb_array(n, data):
    '''Get Nth image from CIFAR data and make 32, 32, 3 np.ndarray 0-255 RGB'''
    arr = np.reshape(data[n], (3, 32, 32))
    return _reorder(arr)


def _reorder(arr):
    '''reorder originial 0-255 (3, x, x) image to (x, x, 3) for PIL/NN'''
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
    ''' take YCC array 0-255
    return RGB array'''
    p = make_pil(np.clip(arr, 0, 255), output_format='RGB')
    return make_arr(p)


def arr_r2y(arr):
    ''' take rgb array 0-255;
    return ycc array'''
    p = make_pil(np.clip(arr, 0, 255), input_format='RGB')
    return make_arr(p)


def show_data(dataset, i=0):
    '''show_data(dataset, i=0) shows the Ith image from +/-1 dataset'''
    im = np.add(np.multiply(dataset[i], 127.5), 127.5)
    make_pil(np.clip(im, 0, 255).astype('uint8'), output_format='RGB').show()


def pil2net(im, scale=127.5):
    '''convert from any PIL image to NN data +/- 1'''
    if im.mode == 'RGB':
        im = im.convert('YCbCr')
    arr = np.empty((1, 32, 32, 3), dtype='float32')
    arr[0] = np.array(im)
    return np.divide(np.subtract(arr, scale), scale)


def net2pil(data, scale=127.5):
    '''make PIL RGB image from neural net +/- 1 data'''
    assert(data.shape == (32, 32, 3))
    arr = np.add(np.multiply(data, scale), scale)
    return make_pil(arr.astype('uint8'), output_format='RGB')


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
