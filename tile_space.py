# tools to tile images into a mosaic (poorly)
import numpy as np

size = (32, 32)


def tileDiagonal(image, imageSet, dim=(3, 3), spc=0):
    '''tileDiagonal takes an image, and tiles out diagonally from imageSet
    until it has filled in the full area defined by dim.ension (size is
    fixed at 32x32) coordinates are (x, y): (1, 1) to (dim[0], dim[1])
    dim is max x, max y'''
    tiled = np.zeros((3, dim[0]*(32+spc), dim[1]*(32+spc)), dtype='float32')
    tiled[:, :32, :32] = image
    pointer = [1, 2]
    mode = 'right'
    for i in range(1, dim[0]*dim[1]):
        print("pointer =", pointer, "  mode =", mode)
        tiled = fitImage(tiled, imageSet, pointer, mode, spc)
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


def fitImage(tiled, imageSet, pointer, mode, spc):
    o = 2   # overlap, set at 1 or more
    oldEdgeB = tiled[:,
                     (pointer[0]-1)*(32+spc)-spc-o:(pointer[0]-1)*(32+spc)-spc,
                     (pointer[1]-1)*(32+spc):pointer[1]*(32+spc)-spc]
    # print('range - bottom edge: row: ', (pointer[0]-1)*32+31,
    #       'cols up to:', pointer[1]*32)
    oldEdgeR = tiled[:,
                     (pointer[0]-1)*(32+spc):pointer[0]*(32+spc)-spc,
                     (pointer[1]-1)*(32+spc)-spc-o:(pointer[1]-1)*(32+spc)-spc]
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
    tiled[:, (pointer[0]-1)*(32+spc):pointer[0]*(32+spc)-spc,
          (pointer[1]-1)*(32+spc):pointer[1]*(32+spc)-spc] = candidate
    return tiled


# fitScore takes 2 edges - np.ndarray shape=(3, 32, 1) or (3, 1, 32)
def fitScore(edge1, edge2):
    # if edge1.shape != edge2.shape:
    #     raise ValueError
    diff = np.subtract(edge1, edge2)
    return np.vdot(diff, diff)
