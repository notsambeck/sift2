# sift generator to replace (random) genycc

import dataset
import numpy as np

# this module will use lasagne-like ordering - [image #, 3, 32, 32]
# quantization matrix determines number of steps for each coefficient

quantization = np.array(range(3172, 100, -1)).reshape((3, 32, 32), order='F')
# placeholder (non-prime) list of quantization values will appear to work

cmax, cmean, cmin, cdist = dataset.loadCifarTransforms()


def getStdDev(imageArray):
    out = np.zeros((3, 32, 32), dtype='float32')
    for row in range(32):
        for col in range(32):
            for chan in range(3):
                out[chan, row, col] = imageArray[:, chan, row, col].std()
    return out

cstd = getStdDev(cdist)
clow = np.subtract(cmean, cstd)


def nextImageWide(count, quantization=quantization):
    transform = np.add(cmin, np.multiply(np.divide(np.subtract(cmax,
                                                               cmin),
                                                   quantization),
                                         count))
    return transform


def nextImageNarrow(count, quantization=quantization):
    transform = np.add(clow, np.multiply(np.divide(np.multiply(cstd, 2.0),
                                                   quantization),
                                         count))
    return transform
