# Datasetfile contains DCT and image conversion tools and utilities to
# load CIFAR dataset to/from a Pickle file.

import numpy as np
import pickle
from math import cos, pi
import matplotlib.pyplot as plt
from PIL import Image

omega = 10000


# LOAD CIFAR TRANSFORMS from pickle made in studyImages:
#
# Max, Mean, Min, Distribution = loadCifarTransforms() This should be
# the only function needed in main app once the pickle is constructed.
def loadCifarTransforms():
    # cifar-10 filename: cifarTransforms.pkl
    # cifar-100 (all 50k):
    pkl_file = open('data/cifarTransforms.pkl', 'rb')
    cifarTransformDistribution = pickle.load(pkl_file)
    cifarMaxTransform = pickle.load(pkl_file)
    cifarMinTransform = pickle.load(pkl_file)
    cifarMeanTransform = pickle.load(pkl_file)
    pkl_file.close()
    return (cifarMaxTransform, cifarMeanTransform,
            cifarMinTransform, cifarTransformDistribution)

# loads pre-pickled dataset of 10k cifar images
# cmax is matrix of maximum values of transform coefs, etc.
cmax, cmean, cmin, cifar = loadCifarTransforms()


# import RGB CIFAR10 batch files from the web of 10k images as
# cifardata, cifarlabels
def importCifar():
    # cifar 10 data in batches 1-5, cifar-100 1 file
    fo = open('data/cifar100.pkl', 'rb')
    u = pickle._Unpickler(fo)
    u.encoding = 'Latin1'
    cifar_data = u.load()
    fo.close()
    cifardata = cifar_data['data']
    print('Imported dataset. Samples:', len(cifardata), ', shape:',
          cifardata.shape)
    return cifardata


# coverts an image from RGB to YCC colorspace
def imR2Y(image):
    out = np.zeros((3, 32, 32), dtype='float32')
    for row in range(32):
        for col in range(32):
            out[:, row, col] = r2y(image[:, row, col])
    return out


# coverts an image from YCC to RGB colorspace 0-255
def imY2R(image):
    out = np.zeros((3, 32, 32), dtype='float32')
    for row in range(32):
        for col in range(32):
            out[:, row, col] = y2rVector(image[:, row, col])
    return out


# coverts an image from YCC to RGB colorspace 0-255
def imY2Rbig(image, resolution=64):
    out = np.zeros((3, resolution, resolution), dtype='float32')
    for row in range(resolution):
        for col in range(resolution):
            out[:, row, col] = y2r(image[:, row, col])
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


# converts to 0-255 RGB pixel for Kivy display
# note that this shoud be vectorized TODO
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
dctMatrix128 = dct_matrix(64)


# DCTII creates YCC transform values from YCC pixel values
def dct(img):
    transform = np.ndarray(img.shape, dtype='float32')
    for i in range(img.shape[0]):
        # print img.shape, dctMatrix.shape, i
        transform[i] = np.dot(np.dot(dctMatrix, img[i]),
                              np.transpose(dctMatrix))
    return transform


# iDCT(II) creates YCC pixel values from YCC transform coefficients
def idct(trans):
    img = np.ndarray(trans.shape, dtype='float32')
    for i in range(trans.shape[0]):
        # print trans.shape, dctMatrix.shape, i
        img[i] = np.dot(np.dot(np.transpose(dctMatrix), trans[i]),
                        dctMatrix)
    return img


# iDCT64(II) creates YCC pixel values from YCC transform coefficients
def idct128(trans):
    img = np.ndarray((3, 64, 64), dtype='float32')
    pad = np.zeros((3, 64, 64), dtype='float32')
    pad[:, :32, :32] = trans
    for i in range(1):
        # print trans.shape, dctMatrix.shape, i
        img[i] = np.dot(np.dot(np.transpose(dctMatrix128), pad[i]),
                        dctMatrix128)
    return img


def chop(trans, compression=1.0):
    for i in range(round(trans.shape[-1]*compression), trans.shape[-1]):
        trans[:, i, :] = 0
        trans[:, :, i] = 0
    return trans


# formTransforms produces statistics about an image dataset (eg CIFAR)
# and a 4-d array of all their transforms for analysis / plotting
# note that this has been run on 10k images (see Load below)
def formTransforms(dataset, numberOfImages=omega):

    # initialize result arrays:
    cifarMaxTransform = np.zeros((3, 32, 32), dtype='float32')
    cifarMinTransform = dct(getImageYCC(0, dataset))
    total = np.zeros((3, 32, 32), dtype='float32')
    # distribution is an arary: RGB transforms stacked numberOfImages deep
    cifarTransformDistribution = np.zeros((numberOfImages, 3, 32, 32),
                                          dtype='float32')
    
    # loop through CIFAR images
    for i in range(numberOfImages):
        transform = dct(getImageYCC(i, dataset))
        cifarMaxTransform = np.maximum(cifarMaxTransform, transform)
        cifarMinTransform = np.minimum(cifarMinTransform, transform)
        total = np.add(total, transform)
        cifarTransformDistribution[i] = transform
        pct = i/numberOfImages*100
        if round(pct) == pct:
            print(''.join([str(pct), '%...']))
    cifarMeanTransform = np.divide(total, numberOfImages)
    out = open('cifarTransforms.pkl', 'wb')
    pickle.dump(cifarTransformDistribution, out)
    pickle.dump(cifarMaxTransform, out)
    pickle.dump(cifarMinTransform, out)
    pickle.dump(cifarMeanTransform, out)
    out.close()


def transformDistance(dataset):
    omega = dataset.shape[0]
    distances = np.array(omega)
    for i in range(omega):
        distances[i] = np.sum(dataset[i]**2)


# quantization = np.array(range(3172, 100, -1),
#                        dtype='float32').reshape((3, 32, 32), order='F')
# placeholder (non-prime) list of quantization values will appear to work


# dataset alternates real/fake...should be OK?
def buildDataset(omega, channels=3, n=4, compression=1.0):
    data = np.zeros((n*omega, channels, 32, 32), dtype='float32')
    label = np.zeros(n*omega, dtype='uint8')
    count = np.zeros((channels, 32, 32), dtype='float32')
    count2 = np.zeros((channels, 32, 32), dtype='float32')
    for i in range(omega):
        pct = 100*i/omega
        if round(pct) == pct:
            print("".join([str(pct), '% ...']))
        count = np.mod(np.add(count, 199.), quantization)
        count2 = np.mod(np.add(count2, 33334.), quantization)
        pos = imY2R(idct(chop(cifar[i], compression)))
        pos2 = imY2R(idct(chop(cifar[i+omega], compression)))
        b = imY2R(idct(chop(nextTransformNarrow(count), compression)))
        c = imY2R(idct(chop(nextTransformNarrow(count2), compression)))
        label[n*i] = 1
        label[n*i+1] = 1
        label[n*i+2] = 0
        label[n*i+3] = 0
        data[n*i] = pos
        data[n*i+1] = pos2
        data[n*i+2] = b
        data[n*i+3] = c
    # data broken up to 90% / 10% to use as desired
    data = np.divide(np.subtract(data, 128.), 128.)
    return (data[0:.9*n*omega], data[.9*n*omega:n*omega],
            label[0:.9*n*omega], label[.9*n*omega:n*omega])


# used 10/15 to save full cifar 100 datatset, should be good to go.
def saveDataset(filename, x, xt, y, yt):
    out = open(filename, 'wb')
    pickle.dump(x, out)
    pickle.dump(xt, out)
    pickle.dump(y, out)
    pickle.dump(yt, out)
    out.close()


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


# reorder an image to 32,32,3 for PIL Image
def orderPIL(image):
    out = np.zeros((32, 32, 3), dtype='uint8')
    for i in range(3):
        out[:, :, i] = image[i]
    return out


# convert from neural net data [(3,32,32) RGB +/- 1] to PIL image
# note that images are NOT normalized to +/- 1
def toPIL(data):
    return Image.fromarray(orderPIL(np.add(np.multiply(data, 128.),
                                           128.)))


def plotHistogram(data, x_range=1, y_range=1, color_range=1):
    # show histograms of transform data (columns in 4th dimension)
    for i in range(x_range):
        for j in range(y_range):
            for k in range(color_range):
                l = (str(i) + " " + str(j) + " " + str(k))
                plt.hist(data[:, k, i, j], 50, label=l)
    plt.legend(loc='upper right')
    plt.show()


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


def getStdDev(imageArray):
    out = np.zeros((3, 32, 32), dtype='float32')
    for row in range(32):
        for col in range(32):
            for chan in range(3):
                out[chan, row, col] = imageArray[:, chan, row, col].std()
    return out

cstd = getStdDev(cifar)
clow = np.subtract(cmean, cstd)


# count neeeds to be float32 dtype
def nextTransformWide(count, quantization=quantization):
    if count.dtype != 'float32':
        raise ValueError(count.dtype)
    transform = np.add(cmin, np.multiply(np.divide(np.subtract(cmax,
                                                               cmin),
                                                   quantization),
                                         count))
    return transform


def nextTransformNarrow(count, quantization=quantization):
    if count.dtype != 'float32':
        raise ValueError(count.dtype)
    transform = np.add(clow, np.multiply(np.divide(np.multiply(2.0,
                                                               cstd),
                                                   quantization),
                                         count))
    return transform
