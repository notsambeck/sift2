# test dataset conversion functions
# also includes histograms, assorted non-core data tools

import dataset
import import_batch

import numpy as np
import matplotlib.pyplot as plt
import timeit
from google.cloud import vision
vc = vision.Client()

test_model = False
if test_model:
    from sift_keras import model, savefile
    model.load_weights(savefile)


def sigma_test(arr1, arr2, test_type, sigma):
    '''test equality of 2 arrays within +/- sigma per element
    takes arr1, arr2, test_type (string for printout only) and sigma
    returns errors: 0 if ~=, 1 if !='''

    if (arr1 == arr2).all():
        print('perfect')
        return(0)

    # if not identical:
    # print(test_type)

    if arr1.shape != arr2.shape:
        print('Arr1 shape={} / Arr2 shape={}'.format(arr1.shape, arr2.shape))
        print('FAIL shapes do not match.')

    else:
        count = 0
        diff = np.subtract(arr1, arr2)
        for ch in range(3):
            for i in range(32):
                for j in range(32):
                    if diff[i, j, ch] > sigma or diff[i, j, ch] < -sigma:
                        count += 1
        if count > 0:
            print('errors found: {} for sigma = {} ?'.format(test_type, sigma))
            print('num. elements outside sigma: {}'.format(count))
            return count
    return 0


def test_dcts():
    x = np.multiply(np.random.rand(32, 32, 3), 255)
    setup = '''
import dataset
import numpy as np
x = np.random.rand(32, 32, 3)
t = dataset.alt_dct(x)
'''
    time1 = timeit.timeit('t1 = dataset.dct(x)', setup=setup, number=1000)
    time2 = timeit.timeit('t2 = dataset.alt_dct(x)', setup=setup, number=1000)
    print('time1: {}, time2: {}'.format(time1, time2))
    t1 = dataset.dct(x)
    t2 = dataset.alt_dct(x)
    print(sigma_test(t1, t2, 'test different dcts', 0.0001))

    itime1 = timeit.timeit('dataset.idct(t)', setup=setup, number=1000)
    itime2 = timeit.timeit('dataset.alt_idct(t)', setup=setup, number=1000)
    print('itime1: {}, itime2: {}'.format(itime1, itime2))
    x1 = dataset.dct(t1)
    x2 = dataset.alt_dct(t1)
    print(sigma_test(x1, x2, 'test different dcts', 0.001))

    return x, t1, t2


# transform / inverse tests
# convert an image to ycc
# transform ycc to tr;
# clip transform coeffs according to nextTransformAdjustable;
# transform back to ycc
# convert back to RGB
# show (optional)

def trinv(data, i, show=1, sigma=20):
    '''trinv is transform-inverse pair test.
    takes: a dataset(4d image stack) and sigma=integer
    returns: sum of errors from sigma_tests of equality of input/output;
    0 if !=, 1 if ~=
    '''

    errors = 0   # counter
    s = sigma    # sigma

    # get rgb from dataset
    arr_in = dataset.get_rgb_array(i, data)  # (32,32,3) ndarray uint8 0-255
    if show > 1:
        print('input array: {}'.format(arr_in.shape))

    # make initial PIL image;
    pil_in = dataset.make_pil(arr_in, input_format='RGB')

    # make keras data:
    # visual way:
    pil_to_net = dataset.pil2net(pil_in)

    # nonvisual way:
    ycc_from_array = dataset.arr_r2y(arr_in)
    if show > 1:
        print('ycc_from_array:')
        print(ycc_from_array[:, :, 0])

    to_net = np.divide(np.subtract(ycc_from_array, 127.5), 127.5)
    as_data = np.empty((1, 32, 32, 3), dtype='float32')
    as_data[0] = to_net
    if show > 1:
        print('min, max = {}, {}'.format(to_net.min(), to_net.max()))
        print('dtype =', to_net.dtype)
        print(to_net[:, :, 0])

    if test_model:
        # run net on originial image
        p = model.predict(as_data)
        if show > 0:
            print(p)
        if p[0, 0] > p[0, 1]:
            print('incorrect prediction on original image')
            errors += 1000

    errors += sigma_test(to_net, pil_to_net, 'two ways to make data', .01)

    # make a ycc version; by default make_arr preserves format
    ycc_from_image = dataset.make_arr(pil_in)
    if show > 1:
        print('ycc arr: dtype = {}, Y channel ='.format(ycc_from_array.dtype))
        print(ycc_from_array[:, :, 0])
    errors += sigma_test(ycc_from_array, ycc_from_image, 'ycc: pil/arr_r2y', s)

    tr = dataset.dct(ycc_from_image)
    if show > 1:
        print('transform: dtype = {}; Y channel ='.format(tr.dtype))
        print(tr[:, :, 0])
    capped = np.clip(tr, dataset.lowest, dataset.highest)
    tr_inv_ycc = dataset.idct(capped)
    if show > 1:
        print('ycc from idct: dtype={}; Y channel ='.format(tr_inv_ycc.dtype))
        print(tr_inv_ycc[:, :, 0])

    errors += sigma_test(tr_inv_ycc, ycc_from_image, 'transform inversion', s)

    to_net2 = np.divide(np.subtract(tr_inv_ycc, 127.5), 127.5)
    as_data2 = np.empty((1, 32, 32, 3), dtype='float32')
    as_data2[0] = to_net2
    if show > 1:
        print('min, max = {}, {}'.format(to_net.min(), to_net.max()))
        print('dtype =', to_net2.dtype)
        print(to_net2[:, :, 0])

    if test_model:
        # run net on capped image
        p2 = model.predict(as_data2)
        if show > 0:
            print(p2)
        if p2[0, 0] > p2[0, 1]:
            print('incorrect prediction on capped image')
            errors += 1000

    im_final = dataset.make_pil(tr_inv_ycc)
    arr_out = dataset.make_arr(im_final, change_format_to='RGB')

    errors += sigma_test(arr_out, arr_in, 'initial vs. final rgb', s)

    from_net_im = dataset.net2pil(to_net)
    from_net_arr = dataset.make_arr(from_net_im, change_format_to='RGB')
    if show > 1:
        print('from_net_arr:')
        print(from_net_arr[:, :, 0])

    errors += sigma_test(from_net_arr, arr_out, 'nn conversion out', s)
    # img_f.show()
    if (errors and show) or show > 1:
        pil_in.show()
        im_final.show()

    return errors


def test_stored_transforms():
    dataset.make_pil(dataset.idct(dataset.cmax)).show()
    dataset.make_pil(dataset.idct(dataset.cmin)).show()
    dataset.make_pil(dataset.idct(dataset.cmean)).show()
    dataset.make_pil(dataset.idct(np.add(dataset.cstd, dataset.cmean))).show()
    dataset.make_pil(dataset.idct(np.subtract(dataset.cstd,
                                              dataset.cmean))).show()


def test_conversions(omega=3):
    print('\n running conversion tests...\n')
    # test import, convert to YCC, transform, revert
    cifar = import_batch.importCifar10(howmany=1)
    print('Testing conversions.')
    errors = 0
    for i in range(omega):
        print(i)
        errors += trinv(cifar, i)

    print('\n TOTAL ERRORS:', errors)
    return cifar


def test_vision():
    cifar = import_batch.importCifar10(howmany=1)
    for i in range(10):
        im = dataset.get_rgb_array(i*100, cifar)
        xim = dataset.expand(im)
        pil = dataset.make_pil(xim, input_format='RGB', output_format='RGB')
        pil.save('.f.png')
        with open('.f.png', 'rb') as f:
            content = f.read()
            goog = vc.image(content=content)
            labels = goog.detect_labels()
        ds = [label.description + str(label.score) for label in labels]
        print(' '.join(ds))


def test_generators(generator_function):
    print('\n testing image generator: {}'.format(generator_function))

    def test_on(count):
        t = generator_function(count)
        im = dataset.make_pil(t, output_format='RGB')
        im.resize((128, 128))
        im.show()

    count = np.zeros((3, 32, 32), dtype='float32')
    test_on(count)
    test_on(dataset.quantization)


# SOME DATA ANALYSIS TOOLS - NOT NECESSARILY USEFUL #

# look at distribution of transforms requires matplotlib import
def plotHistogram(data, x_range=2, y_range=2, colors=3):
    # show histograms of transform data (columns in 4th dimension)
    for i in range(x_range):
        for j in range(y_range):
            for k in range(colors):
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


if __name__ == '__main__':
    print('import cifar, test on i=100 images')
    test_conversions(100)
