# function library for tiny image dataset
# https://github.com/cioc/PyTinyImage/blob/master/tinyimage.py
import numpy as np

# paths to various data files
# meta_file_path = "/tiny/tinyimages/tiny_metadata.bin"
data_file_path = "../tiny_images/tiny_images.bin"

img_count = 79302017


def load_tiny(n=0):
    with open(data_file_path, 'rb') as f:
        # imgs = np.empty((n, 32, 32, 3))
        f.seek(3072*n)
        return np.fromstring(f.read(3072), dtype='uint8')\
                 .reshape((32, 32, 3), order='F')
