import os
import pickle
import numpy as np
from PIL import Image

import dataset

size = (32, 32)


def getImages(filepath):
    all_files = os.listdir(filepath)
    # change 1 to 2 in following line if there are non-square images:
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


# tileDiagonal takes an image, and tiles out diagonally from imageSet
# until it has filled in the full area defined by dim.ension (size is
# fixed at 32x32) coordinates are (x, y): (1, 1) to (dim[0], dim[1])
# dim is max x, max y
def tileDiagonal(image, imageSet, dim=(3, 3)):
    tiled = np.zeros((3, dim[0]*32, dim[1]*32), dtype='float32')
    tiled[:, :32, :32] = image
    pointer = [1, 2]
    mode = 'right'
    for i in range(1, dim[0]*dim[1]):
        print("pointer =", pointer, "  mode =", mode)
        tiled = fitImage(tiled, imageSet, pointer, mode)
        # if at left edge and not in bottom row, bounce to top
        if pointer[1] == 1 and pointer[0] < dim[0]:
            pointer[1] = pointer[0] + 1
            pointer[0] = 1
            mode = 'right'
        elif pointer[0] == dim[0]:  # if at bottom row, bounce to right edge
            pointer[0] = pointer[1] + 1
            pointer[1] = dim[1]
            mode = 'both'
        else:  # else move diagonally
            pointer[1] -= 1
            pointer[0] += 1
            if pointer[1] == 1:
                mode = 'below'
            else:
                mode = 'both'
    return tiled


def fitImage(tiled, imageSet, pointer, mode):
    o = 2   # overlap, set at 1 or more
    oldEdgeB = tiled[:,
                     (pointer[0]-1)*32-o:(pointer[0]-1)*32,
                     (pointer[1]-1)*32:pointer[1]*32]
    # print('range - bottom edge: row: ', (pointer[0]-1)*32+31,
    #       'cols up to:', pointer[1]*32)
    oldEdgeR = tiled[:,
                     (pointer[0]-1)*32:pointer[0]*32,
                     (pointer[1]-1)*32-o:(pointer[1]-1)*32]
    # print('range - right  edge: rows up to:', pointer[0]*32,
    #       'column:', (pointer[1]-1)*32+31)
    bestScore = 10.**10
    candidate = np.zeros((3, 32, 32), dtype='uint8')
    candidateNumber = 0
    for i in range(imageSet.shape[0]):
        if mode == 'below':
            score = fitScore(imageSet[i, :, :o, :], oldEdgeB)
        elif mode == 'right':
            score = fitScore(imageSet[i, :, :, :o], oldEdgeR)
        elif mode == 'both':
            score = (fitScore(imageSet[i, :, :o, :], oldEdgeB) +
                     fitScore(imageSet[i, :, :, :o], oldEdgeR))
        else:
            raise IOError("invalid mode")

        if score < bestScore:
            candidate = imageSet[i]
            bestScore = score
            # print(score)
            candidateNumber = i  # if you want to remove from set
    tiled[:, (pointer[0]-1)*32:pointer[0]*32,
          (pointer[1]-1)*32:pointer[1]*32] = candidate
    return tiled


# fitScore takes 2 edges - np.ndarray shape=(3, 32, 1) or (3, 1, 32)
def fitScore(edge1, edge2):
    # if edge1.shape != edge2.shape:
    #     raise ValueError
    diff = np.subtract(edge1, edge2)
    return np.vdot(diff, diff)
