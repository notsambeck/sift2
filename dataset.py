# Dataset.py contains DCT and image conversion tools and utilities to
# load CIFAR dataset, analyze it, and create images.
# Also contains low-level image transform and retrieval tools.

import numpy as np
import pickle
from math import cos, pi
from PIL import Image

# import_batch is a SIFT toolset for loading pickled sets of images
# useful for testing and retraining network:
# import import_batch

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
    fo = open('data/cifar_raw_data/cifar100.pkl', 'rb')
    u = pickle._Unpickler(fo)
    u.encoding = 'Latin1'
    cifar_data = u.load()
    fo.close()
    cifar100 = cifar_data['data']
    print('Imported dataset. Samples:', len(cifar100), ', shape:',
          cifar100.shape)
    return cifar100


# import RGB CIFAR10 batch files of 10k images
# return 3072 x n np.array in range 0-255
# NEEDS TO BE SCALED to +/- 1
def importCifar10(howmany=1):
    # cifar 10 data in batches 1-5
    cifar10 = np.zeros((50000, 3072), dtype='uint8')
    for i in range(1, howmany+1):
        name = ''.join(['data/cifar_raw_data/data_batch_', str(i)])
        f = open(name, 'rb')
        u = pickle._Unpickler(f)
        u.encoding = 'Latin1'
        cifar_data = u.load()
        f.close()
        cifar10[(i-1)*10000:i*10000] = cifar_data['data']
    print('Imported dataset.')
    print('Samples: {}, shape: {}, range: {} to {}.'.format(len(cifar10),
                                                            cifar10.shape,
                                                            cifar10.min(),
                                                            cifar10.max()))
    return cifar10


# coverts an image from RGB to YCC colorspace
def imR2Y(image):
    out = np.zeros((3, 32, 32), dtype='float32')
    for row in range(32):
        for col in range(32):
            out[:, row, col] = r2y(image[:, row, col])
    return out


# coverts an image from YCC to RGB colorspace 0-255
def imY2R(image):
    out = np.zeros(image.shape, dtype='float32')
    for row in range(image.shape[-1]):
        for col in range(image.shape[-1]):
            out[:, row, col] = y2rVector(image[:, row, col])
    return out


# matrices for colorspace conversion #
r2yconversion = np.transpose(np.array([[65.738, 129.057, 25.064],
                                       [-37.945, -74.494, 112.439],
                                       [112.439, -94.154, -18.285]]))
r2yaddition = np.array([16., 128., 128.])


# converts a pixel from RGB to YCbCr colorspace using above matrices #
def r2y(pixel):
    YCC = np.dot(np.divide(pixel, 256.), r2yconversion)
    return np.add(YCC, r2yaddition)


y2rMult = np.transpose(np.array([[298.082, 0., 408.583],
                                 [298.082, -100.291, -208.120],
                                 [298.082, 516.412, 0]]))
y2rAdd = np.array([-222.921, 135.576, -276.836])


# y2r vector implements same conversion as below but faster
# leaving old version as a reference
def y2rVector(YCCpixel):
    rgb = np.add(np.dot(np.divide(YCCpixel, 256.), y2rMult), y2rAdd)
    np.clip(rgb, 0, 255, out=rgb)
    return rgb


# converts to 0-255 RGB pixel for display
def y2r(pixel):
    Y, Cb, Cr = np.divide(pixel, 256.)
    R = max(min(255, 298.082 * Y + 408.583 * Cr - 222.921), 0)
    G = max(min(255, 298.082 * Y - 100.291 * Cb - 208.120 * Cr + 135.576), 0)
    B = max(min(255, 298.082 * Y + 516.412 * Cb - 276.836), 0)
    return np.array([R, G, B])


# DCT TOOLS SECTION #


# create DCT matrix for n x n image
def dct_matrix(n):
    d = np.ones((n, n), dtype='float32')
    d[0, :] = np.multiply(d[0, :], 1./(n**.5))
    for row in range(1, n):
        for col in range(n):
            d[row][col] = (2./n)**.5*cos(row*pi*(2.*col+1.)/2./n)
    return(d)


dctMatrix = dct_matrix(32)
bigSize = 512
dctMatrixBig = dct_matrix(bigSize)


# DCTII creates YCC transform values from YCC pixel values
def dct(img):
    transform = np.ndarray(img.shape, dtype='float32')
    for i in range(img.shape[0]):
        # print img.shape, dctMatrix.shape, i
        transform[i] = np.dot(np.dot(dctMatrix, img[i]),
                              np.transpose(dctMatrix))
    return transform


# iDCT(DCT III) creates YCC pixel values from YCC transform coefficients
def idct(trans):
    img = np.ndarray(trans.shape, dtype='float32')
    for i in range(trans.shape[0]):
        img[i] = np.dot(np.dot(np.transpose(dctMatrix), trans[i]),
                        dctMatrix)
    return img


def idctBig(trans):
    bigtrans = np.zeros((3, bigSize, bigSize), dtype=trans.dtype)
    bigtrans[:, :32, :32] = trans
    img = np.ndarray((3, bigSize, bigSize), dtype='float32')
    for i in range(3):
        img[i] = np.dot(np.dot(np.transpose(dctMatrixBig), bigtrans[i]),
                        dctMatrixBig)
    return img


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
        out[i] = nextTransformNarrow(count)
    return out


# this was a float32 out, but I beleive that caused the neon blobs?
def buildRGBIncremented(omega, inc=44777):
    out = np.zeros((omega, 3, 32, 32), dtype='uint8')
    count = np.zeros((3, 32, 32), dtype='float32')
    for i in range(omega):
        count = np.mod(np.add(count, inc), quantization)
        out[i] = imY2R(idct(nextTransformNarrow(count)))
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
    return(out)


def generateRGBs(n, maximum, minimum):
    randomimages = np.zeros((omega, 3, 32, 32), 'uint8')
    for i in range(n):
        temp = genycc(maximum, minimum)
        # if desired: print("encoding image ", i, temp[0, 0])
        ycc = idct(temp)
        rgb = imY2R(ycc)
        randomimages[i] = rgb
    return randomimages


# pull an image from CIFAR data and make  32, 32, 3 numpy.array 0-255 RGB
def getImage255(n, dataset):
    image = np.reshape(dataset[n], (3, 32, 32))
    return(image)


# takes an integer, returns that image from CIFAR in YCC format
def getImageYCC(n, dataset):
    return(imR2Y(getImage255(n, dataset)))


# reorder an 0-255 (3, x, x) image to (x, x, 3) for PIL
def orderPIL(image):
    out = np.zeros((image.shape[1], image.shape[2], 3),
                   dtype='uint8')
    for i in range(3):
        out[:, :, i] = image[i]
    return out


# convert from neural net data [(3,32,32) RGB +/- 1] to PIL image
# call only on NN data, not images
def toPIL(data, scale=128):
    return Image.fromarray(orderPIL(np.add(np.multiply(data, scale),
                                           scale)))


# SOME DATA ANALYSIS TOOLS - NOT NECESSARILY USEFUL #

# look at distribution of transforms requires matplotlib import
def plotHistogram(data, x_range=1, y_range=1, color_range=1):
    # show histograms of transform data (columns in 4th dimension)
    for i in range(x_range):
        for j in range(y_range):
            for k in range(color_range):
                l = (str(i) + " " + str(j) + " " + str(k))
                plt.hist(data[:, k, i, j], 50, label=l)
    plt.legend(loc='upper right')
    plt.show()


# take a matrix and list it in ~order of significance of coefs.
def diagonalUnfold(image, channels=1):
    vector = np.zeros((channels, 1024))
    for i in range(channels):
        count = 0
        for j in range(32):
            for k in range(j+1):
                vector[i, count] = image[i, j-k, k]
                count += 1
        for j in range(31):
            for k in range(j+1):
                vector[i, count] = image[i, 30-k, k]
                count += 1
    return vector


# SIFT IMAGE (Transform) GENERATOR FUNCTIONS! #

# all of these are functional, but nextTransformAdjustable is used
# in SIFT for aesthetic reasons.


# narrowScale determines the range of transforms created below;
# larger values for narrowScale create more contrast-y images.
# 1.6 seems to be about right but this is totally subjective.
narrowScale = 1.6

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


# convert an image to ycc
# transform ycc to tr;
# clip transform coeffs according to nextTransformAdjustable;
# transform back to ycc
# convert back to RGB
# show
def trinv(dataset, i=0):
    img255 = getImage255(i, dataset)

    show = np.divide(np.subtract(img255, 128), 128)  # +-1
    toPIL(show).show()  # show image +/- 1

    ycc255 = imY2R(img255)
    img255 = imR2Y(ycc255)
    show = np.divide(np.subtract(img255, 128), 128)  # +-1
    toPIL(show).show()  # show afer transform

    tr = dct(ycc255)
    r = np.multiply(3.2, cstd)
    high = np.add(clow, r)
    capped = np.maximum(clow, tr)
    capped = np.minimum(high, capped)
    ycc = idct(capped)
    im255 = imY2R(ycc)
    img = np.divide(np.subtract(im255, 128), 128)
    toPIL(img).show()
    return img


# create next image in full range - min to max.
# leads to a lot of blacks and whites, narrow is subjectively better
def nextTransformWide(count, quantization=quantization):
    if count.dtype != 'float32':
        raise ValueError(count.dtype)
    transform = np.add(cmin, np.multiply(np.divide(np.subtract(cmax,
                                                               cmin),
                                                   quantization),
                                         count))
    return transform


# count neeeds to be float32 dtype!
# CREATE NEXT IMAGE USING A COUNTER MATRIX
# narrowScale 1.6  matches distribution of real images OK...
def nextTransformNarrow(count):
    if count.dtype != 'float32':
        raise ValueError(count.dtype)
    return np.add(clow, np.multiply(count, cmult))


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
