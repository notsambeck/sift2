# Datasetfile contains DCT and image conversion tools and utilities to
# load CIFAR dataset to/from a Pickle file.

import numpy as np
import pickle
from math import cos, pi
import matplotlib.pyplot as plt

omega = 2000
imageSize = 32


# LOAD CIFAR TRANSFORMS from pickle made in studyImages:
#
# Max, Mean, Min, Distribution = loadCifarTransforms() This should be
# the only function needed in main app once the pickle is constructed.
def loadCifarTransforms():
    pkl_file = open('cifarTransforms.pkl', 'rb')
    cifarTransformDistribution = pickle.load(pkl_file)
    cifarMaxTransform = pickle.load(pkl_file)
    cifarMinTransform = pickle.load(pkl_file)
    cifarMeanTransform = pickle.load(pkl_file)
    pkl_file.close()
    return (cifarMaxTransform, cifarMeanTransform,
            cifarMinTransform, cifarTransformDistribution)


# import RGB CIFAR10 batch files from the web of 10k images as
# cifardata, cifarlabels
def importCifar():
    for i in range(1, 2):
        fo = open('data_batch_'+str(i)+'.pkl', 'rb')
        u = pickle._Unpickler(fo)
        u.encoding = 'Latin1'
        cifar_data = u.load()
        fo.close()
    cifardata = cifar_data['data']
    cifarlabels = cifar_data['labels']
    return(cifardata, cifarlabels)


# coverts an image from RGB to YCC colorspace
def imR2Y(image):
    out = np.zeros((3, 32, 32), dtype='float32')
    for row in range(32):
        for col in range(32):
            out[:, row, col] = r2y(image[:, row, col])
    return out


# coverts an image from YCC to RGB colorspace
def imY2R(image):
    out = np.zeros((3, 32, 32), dtype='float32')
    for row in range(32):
        for col in range(32):
            out[:, row, col] = y2r(image[:, row, col])
    return out


def im2vec(image):
    return image.reshape((1, -1))


def vec2im(vector):
    return vector.reshape((3, 32, 32))

# matrices for colorspace conversion #
conversion = np.transpose(np.array([[65.738, 129.057, 25.064],
                                    [-37.945, -74.494, 112.439],
                                    [112.439, -94.154, -18.285]]))
addition = np.array([16., 128., 128.])


# converts a pixel from RGB to YCC colorspace using above matrices #
def r2y(pixel):
    YCC = np.dot(np.divide(pixel, 256.), conversion)
    return np.add(YCC, addition)


# converts to 0-255 RGB pixel for Kivy display
def y2r(pixel):
    Y, Cb, Cr = np.divide(pixel, 256.)
    R = 298.082 * Y + 408.583 * Cr - 222.921
    G = 298.082 * Y - 100.291 * Cb - 208.120 * Cr + 135.576
    B = 298.082 * Y + 516.412 * Cb - 276.836
    return np.array([R, G, B])


# DCT TOOLS SECTION #


# create DCT matrix for n x n image
def dct_matrix(n):
    d = np.ones((n, n))
    d[0, :] = np.multiply(d[0, :], 1./(n**.5))
    for row in range(1, n):
        for col in range(n):
            d[row][col] = (2./n)**.5*cos(row*pi*(2.*col+1.)/2./n)
    return(d)


dctMatrix = dct_matrix(imageSize)


# DCTII creates transform values from pixel values
def dct(img):
    transform = np.ndarray(img.shape)
    for i in range(img.ndim):
        # print img.shape, dctMatrix.shape, i
        transform[i] = np.dot(np.dot(dctMatrix, img[i]),
                              np.transpose(dctMatrix))
    return transform


# iDCT(II) creates pixel values from transform coefficients
def idct(trans):
    img = np.ndarray(trans.shape)
    for i in range(trans.ndim):
        # print trans.shape, dctMatrix.shape, i
        img[i] = np.dot(np.dot(np.transpose(dctMatrix), trans[i]),
                        dctMatrix)
    return img


def chop(trans, compression=1.0):
    for i in range(round(trans.shape[-1]*compression), trans.shape[-1]):
        trans[:, i, :] = 0
        trans[:, :, i] = 0
    return trans


# studyImages produces statistics about an image dataset (eg CIFAR)
# and a 4-d array of all their transforms for analysis / plotting
# note that this has been run on 10k images (see Load below)
def studyImages(dataset, numberOfImages=omega):

    # initialize result arrays:
    cifarMaxTransform = np.zeros((3, 32, 32), float)
    cifarMinTransform = np.multiply(np.ones((3, 32, 32), float), 100000)
    total = np.zeros((3, 32, 32), float)
    # distribution is an arary: RGB transforms stacked numberOfImages deep
    cifarTransformDistribution = np.zeros((numberOfImages, 3, 32, 32))

    # loop through CIFAR images
    for i in range(numberOfImages):
        transform = dct(getImageYCC(i, dataset))
        cifarMaxTransform = np.maximum(cifarMaxTransform, transform)
        cifarMinTransform = np.minimum(cifarMinTransform, transform)
        total = np.add(total, transform)
        cifarTransformDistribution[i] = transform
    cifarMeanTransform = np.divide(total, numberOfImages)
    out = open('cifarTransforms.pkl', 'wb')
    pickle.dump(cifarTransformDistribution, out)
    pickle.dump(cifarMaxTransform, out)
    pickle.dump(cifarMinTransform, out)
    pickle.dump(cifarMeanTransform, out)
    out.close()


def genycc(transformMax, transformMin, chopPoint=32, center=.5, sigma=.01):
    # generates YCC transform according to normal distribution
    # between two limit matrices generated from CIFAR
    out = np.zeros((3, 32, 32))  # output will be a RGB array
    rando = np.random.normal(loc=center, scale=sigma,
                             size=(3, 32, 32))
    out = np.add(transformMin,
                 np.multiply(np.subtract(transformMax, transformMin),
                             rando))
    for i in range(chopPoint, imageSize):
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


def pillify(image):
    out = np.zeros((32, 32, 3), dtype='uint8')
    for i in range(3):
        out[:, :, i] = image[i]
    return out


def plotHistogram(data, x_range=1, y_range=1, color_range=1):
    # show histograms of transform data (columns in 4th dimension)
    for i in range(x_range):
        for j in range(y_range):
            for k in range(color_range):
                l = (str(i) + " " + str(j) + " " + str(k))
                plt.hist(data[:, i, j, k], 50, label=l)
    plt.legend(loc='upper right')
    plt.show()
