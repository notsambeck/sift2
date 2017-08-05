# test dataset conversion functions
# also includes histograms, assorted non-core data tools

import numpy as np
import matplotlib.pyplot as plt
import dataset


def sigma_test(arr1, arr2, test_type, sigma):
    '''test equality of 2 arrays within +/- sigma per element
    takes arr1, arr2, test_type (string for printout only) and sigma
    returns 1 if ~=, 0 if !='''

    if arr1.all() == arr2.all():
        return(0)

    # if not identical:
    # print(test_type)

    if arr1.shape != arr2.shape:
        print('Arr1 shape={} / Arr2 shape={}'.format(arr1.shape, arr2.shape))
        print('FAIL shapes do not match.')

    else:
        count = 0
        diff = np.subtract(arr1, arr2)
        for d in range(3):
            for i in range(32):
                for j in range(32):
                    if diff[d][i][j] > sigma or diff[d][i][j] < -sigma:
                        count += 1
        if count > 1:
            print('FAIL {} for sigma = {} ?'.format(test_type, sigma))
            print('#elements outside sigma: {}'.format(count))
            return 1
    return 0


# transform / inverse tests
# convert an image to ycc
# transform ycc to tr;
# clip transform coeffs according to nextTransformAdjustable;
# transform back to ycc
# convert back to RGB
# show (optional)

def trinv(data, i, show=True, sigma=10):
    '''trinv is transform-inverse.
    takes: a dataset(4d image stack) and an integer
    returns: sigma_test of equality of input/output;
    0 if !=, 1 if ~=
    parameter s is sigma (internal)'''

    errors = 0   # counter
    s = sigma   # sigma

    # get rgb from dataset
    arr_in = dataset.get_rgb_array(i, data)  # (3,32,32) ndarray uint8 0-255

    # make initial PIL image;
    pil_in = dataset.make_pil(arr_in, input_format='RGB')

    # make a ycc version; by default make_arr preserves format
    ycc_from_image = dataset.make_arr(pil_in)
    ycc_from_array = dataset.arr_r2y(arr_in)
    errors += sigma_test(ycc_from_array, ycc_from_image, 'ycc: pil/arr_r2y', s)

    tr = dataset.dct(ycc_from_image)
    capped = np.clip(tr, dataset.lowest, dataset.highest)
    tr_inv_ycc = dataset.idct(capped)

    errors += sigma_test(tr_inv_ycc, ycc_from_image, 'transform inversion', s)

    im_final = dataset.make_pil(tr_inv_ycc)
    arr_out = dataset.make_arr(im_final, change_format_to='RGB')

    errors += sigma_test(arr_out, arr_in, 'initial vs. final rgb', s)
    # img_f.show()
    if errors and show:
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


def test_conversions(omega=10):
    print('\n running conversion tests...\n')
    # test import, convert to YCC, transform, revert
    cifar = dataset.importCifar10()
    print('Testing conversions.')
    errors = 0
    for i in range(omega):
        print(i)
        errors += trinv(cifar, i)

    print('\n TOTAL ERRORS:', errors)


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
