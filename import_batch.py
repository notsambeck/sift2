import os
import pickle
import numpy as np
from PIL import Image

import dataset

size = (32, 32)


def getImages(filepath):
    all_files = os.listdir(filepath)
    out = np.zeros((len(all_files)*2, 3, 32, 32), dtype='float32')
    count = 0

    def addToDataset(image):
        if image.size != size:
            image = image.resize(size)
        p = np.array(image)
        for j in range(3):
            # print('range,', j, 'ok')
            out[count, j] = p[:, :, j]

    for file in all_files:
        # print('file ok:', file)
        i = Image.open(''.join([filepath, file]))
        if i.size[0] >= round(1.5*i.size[1]):
            i1 = i.crop((10, 10, i.size[1]-10, i.size[1]-10))
            addToDataset(i1)
            count += 1
            i2 = i.crop((i.size[0]-i.size[1]+10, 10,
                         i.size[0]-10, i.size[1]-10))
            addToDataset(i2)
            count += 1
        elif i.size[1] >= round(1.5*i.size[0]):
            i1 = i.crop((10, 10, i.size[0]-10, i.size[0]-10))
            addToDataset(i1)
            count += 1
            i2 = i.crop((10, i.size[1]-i.size[0]+10,
                         i.size[0]-10, i.size[1]-10))
            addToDataset(i2)
            count += 1
        else:
            addToDataset(i)
            count += 1

    out = np.subtract(out, 128)
    out = np.divide(out, 128)
    print('images:', count, 'of', len(all_files))
    print(out.max(), '= MAX, MIN =', out.min(), 'MEAN =', out.mean())
    return out[:count]
