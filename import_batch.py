import os
import pickle
import numpy as np
from PIL import Image


def getImages(filepath):
    all_files = os.listdir(filepath)
    out = np.zeros((len(all_files), 3, 32, 32), dtype='float32')
    count = 0
    for file in all_files:
        i = Image.open(''.join([filepath, file]))
        i.resize((32, 32), Image.ANTIALIAS)
        p = np.array(i)
        for j in range(3):
            out[count, j] = p[:, :, j]
        count += 1
    return out
