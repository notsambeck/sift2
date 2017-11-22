from sift_keras import savefile, predict_incorrect
import dataset as ds
import os
import kivy
import numpy as np


def test_requirements_installed():
    assert kivy.exists('.')


def test_load_weights():
    assert os.path.exists(savefile)


def test_model_preds():
    testfile = 'data/portable_test_dataset_0.pkl'
    assert predict_incorrect(filename=testfile, limit=1, output=False)
    print('model accuracy test passed! <= 1 errors of 1000 test images')


class TestDatasetTools(object):
    # create a transform using pickled stats; do stuff to it.
    def gen_transform(self):
        self.tr = ds.random_transform(mean=ds.cmean,
                                      std_dev=ds.cstd)

    def test_transform_valid(self):
        self.gen_transform()
        assert self.tr.shape == (32, 32, 3)
        assert self.tr.dtype == 'float32'

    def test_conversion(self):
        errors = []
        for i in range(10):
            self.gen_transform()
            # make ycc image from transform:
            ycc_in = ds.idct(self.tr)

            # make pil image, show it
            pil1 = ds.make_pil(ycc_in, input_format='YCbCr',
                               output_format='RGB')
            # print(ycc_in.max(), ycc_in.min(), ycc_in[0:2, 0:2], ycc_in.shape)
            # pil1.show()

            # make new array as RGB
            rgb_arr = ds.make_arr(pil1)

            # convert to ycc
            ycc_out = ds.arr_r2y(rgb_arr)
            # print(ycc_out.max(), ycc_out.min(), ycc_out[0:2, 0:2])

            x = np.subtract(ycc_out.astype('float32'),
                            ycc_in.astype('float32')).mean()

            errors.append(x ** 2)
            assert -1 < np.mean(errors) < 1
