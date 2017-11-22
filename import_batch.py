import os
import numpy as np
from PIL import Image
import dataset
import pickle


class Dataset():
    '''Dataset is an np.ndarray, shape = (omega, 32, 32, 3)
    Images are stored in YCbCr format, scaled to +/- 1 values.
    '''
    def __init__(self, omega, shape=(32, 32, 3), size=(32, 32)):
        self.data = np.empty((omega, shape[0], shape[1], shape[2]),
                             dtype='float32')
        self.labels = np.empty(omega, dtype='uint8')
        self.omega = omega
        self.count = 0
        self.shape = shape
        self.size = size
        self.full = False
        self.just_shuffled = False
        self.display = 5   # show self.display samples from each action

    def __str__(self):
        return 'Dataset of len {} filled to {}'.format(self.omega, self.count)

    def __repr__(self):
        return str(self)

    def _ftd_add(self, frmtd, label):
        ''' adds PIL image to dataset if formatted, do not call directly'''
        if self.full:
            print('sorry, dataset is full')
            return False
        if frmtd.size != self.size:
            raise ValueError('image not correctly formatted for size')
        if frmtd.mode != 'YCbCr':
            frmtd = frmtd.convert('YCbCr')
        try:
            self.data[self.count] = dataset.pil2net(frmtd)[0]  # pil2net is 4d
            self.labels[self.count] = label
            self.count += 1
            self.just_shuffled = False
        except:
            ValueError('dimensions wrong; skip')
            print(frmtd)
        if self.count == self.omega:
            print('warning! at capacity')
            self.full = True

    def add(self, image, label):
        '''adds a PIL image to dataset'''
        # size conditionals
        if image.size == self.size:
            self._ftd_add(image, label)
        elif image.size[0] < 32 or image.size[1] < 32:
            print('too small; skip')
            return False
        elif image.size[0] < 64 or image.size[1] < 64:
            self._ftd_add(image.resize((32, 32)))
        else:
            self._fragment_add(image, label)

    def _fragment_add(self, image, label):
        '''breaks up a large image, adds to dataset '''
        w, h = image.size
        if image.mode != 'YCbCr':
            image = image.convert('YCbCr')
        if w >= 1.3 * h:    # horizontal
            cols = 6
            rows = 4
        elif h >= 1.3 * w:  # vertical
            cols = 4
            rows = 6
        else:
            cols = 4
            rows = 4
        image = image.resize((cols * 32, rows * 32))
        for r in range(rows):
            for c in range(cols):
                orig = image.crop((c*32, r*32, (c + 1)*32,
                                   (r + 1)*32))
                capped = self._cap(orig)
                self._ftd_add(orig, label)
                self._ftd_add(capped, label)

    def _cap(self, image):
        '''takes a PIL YCC image; returns that same image capped'''
        if image.mode != 'YCbCr':
            raise ValueError(image.mode)
        ycc_arr = dataset.make_arr(image)
        tr = dataset.dct(ycc_arr)
        capped_tr = np.clip(tr, dataset.lowest, dataset.highest)
        capped_ycc_arr = np.clip(dataset.idct(capped_tr), 0, 255)
        return dataset.make_pil(capped_ycc_arr)

    def add_dir(self, filepath, label):
        '''takes a directory and adds a stack of PIL images to data
        automatically adds capped & non-capped versions for non-32x32 images
        label = 1 for image, 0 for non-image'''
        all_files = os.listdir(filepath)
        print('dir contains {} files, label={}'.format(len(all_files), label))
        files_added = 0
        o = self.count
        for f in all_files:
            print('{} %'.format(files_added/len(all_files)*100), end='\r')
            if self.full:
                break
            if f[-4:] in ['.png', '.jpg', '.PNG', '.JPG', 'JPEG']:
                image = Image.open('/'.join([filepath, f]))
                self.add(image, label)
                files_added += 1
        print('loaded {} files of {} objs; =>{} images'.format(files_added,
                                                               len(all_files),
                                                               self.count - o))
        self.visual_test()

    def add_cifar(self, qty=100000):
        '''
        adds 2 * qty CIFAR images to dataset - both capped & original versions
        '''
        print('loading cifar datasets...')
        cifar = np.append(importCifar10(), importCifar100(), axis=0)
        print('done')
        for i in range(qty):
            if i % 1000 == 0:
                print('{} %'.format(i/qty*100), end='\r')
            rgb = dataset.get_rgb_array(i, cifar)
            ycc = dataset.make_pil(rgb, input_format='RGB')
            capped = self._cap(ycc)
            if i % 2 == 1:
                ycc = ycc.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                capped = capped.transpose(Image.FLIP_LEFT_RIGHT)
            self.add(ycc, 1)
            self.add(capped, 1)
        self.visual_test()

    def add_tiny_images(self, qty=100000):
        '''ads qty images from tiny image dataset'''
        for i in range(qty):
            rgb = load_tiny(i)
            ycc = dataset.make_pil(rgb, 'RGB')
            capped = self._cap(ycc)
            self.add(capped, 1)
        self.visual_test()

    def add_generated(self, qty, increment=999):
        '''generates and adds transforms'''
        count = np.zeros((self.shape), dtype='float32')
        for i in range(qty):
            count = np.add(count, increment)
            count = np.mod(count, dataset.quantization)
            ycc = dataset.idct(dataset.get_transform(count))
            self.add(dataset.make_pil(ycc), 0)  # label is zero
            if i % (qty // 100) == 0:
                print('{} %'.format(i * 100 / qty), end='\r')
        self.visual_test()

    def random_levels(self):
        self.visual_test()
        input('press enter to randomize levels')
        for e in self.data:
            e = np.multiply(1 + np.random.randn()/4, e)
            np.clip(e, -1, 1, out=e)
        self.visual_test()

    def shuffle(self):
        # shuffle data
        state = np.random.get_state()
        np.random.shuffle(self.data[:self.count])
        np.random.set_state(state)
        np.random.shuffle(self.labels[:self.count])
        self.just_shuffled = True
        self.visual_test()

    def visual_test(self):
        '''show last self.display images in data'''
        # show self.display images unless not that many are available.
        display = min(self.count, self.display)
        for i in range(self.count - display, self.count):
            dataset.show_data(self.data, i)
            print(i, '=>', self.labels[i])

    def save(self, filename, chunks=10, confirm=True):
        '''
        Enter a simple name for your dataset; save writes it to data directory
        as #(chunks) .pkl files. Extension/location are added automatically.
        '''
        if '.' in filename:
            raise NameError('Use simple name, data/{filename}_0.pkl is added.')
        # data broken into chunks:
        if confirm and not self.just_shuffled:
            print('dataset not shuffled')
            return False
        print('min = {}, max = {}'.format(self.data.min(), self.data.max()))
        chunk_size = self.count // chunks
        print('will write {} chunks of len({}) out of {}'.format(chunks,
                                                                 chunk_size,
                                                                 self.count))
        for chunk in range(chunks):
            filepath = ''.join(['data/', filename, '_', str(chunk), '.pkl'])
            print('writing pkl to {}'.format(filepath))
            with open(filepath, 'wb') as f:
                pickle.dump(self.data[chunk*chunk_size:(chunk+1)*chunk_size],
                            f)
                pickle.dump(self.labels[chunk*chunk_size:(chunk+1)*chunk_size],
                            f)


def combine_data_labels(x1, x2, y1, y2):
    if x1.max() != x2.max():
        raise ValueError('different dtypes in data to combine')
    x = np.zeros((x1.shape[0]+x2.shape[0], 3, 32, 32), dtype='float32')
    y = np.zeros(x1.shape[0]+x2.shape[0], dtype='uint8')
    x[:x1.shape[0]] = x1
    x[x1.shape[0]:] = x2
    y[:x1.shape[0]] = y1
    y[x1.shape[0]:] = y2
    state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(state)
    np.random.shuffle(y)
    return x, y


def make_transforms(data='cifar'):
    '''makeTransforms produces statistics about an image dataset (eg CIFAR)
    and a 4-d array of all their transforms for analysis / plotting.
    once assembled, you can load this pickle instead of rebuilding it
    '''
    shape = (32, 32, 3)
    if data == 'cifar':
        data = np.concatenate((importCifar10(), importCifar100()), axis=0)
    print(data.shape)

    # initialize result arrays:
    cifarMaxTransform = np.multiply(np.ones(shape, dtype='float32'), -100000)
    cifarMinTransform = np.multiply(np.ones(shape, dtype='float32'), 100000)
    total = np.zeros(shape, dtype='float32')

    # format: RGB transforms stacked numberOfImages deep
    cifarTransforms = np.zeros((len(data), 32, 32, 3),
                               dtype='float32')

    # loop through CIFAR images
    for i in range(len(data)):
        rgb = dataset.get_rgb_array(i, data)
        ycc = dataset.arr_r2y(rgb)
        transform = dataset.dct(ycc)
        cifarMaxTransform = np.maximum(cifarMaxTransform, transform)
        cifarMinTransform = np.minimum(cifarMinTransform, transform)
        total = np.add(total, transform)
        cifarTransforms[i] = transform
        pct = i/len(data)*100
        if round(pct) == pct:
            print('{} %'.format(pct), end='\r')
    cifarMeanTransform = np.divide(total, len(data))
    with open('init_data', 'wb') as out:
        cstd = getStdDev(cifarTransforms)

        # pickle.dump(cifarTransforms, out)    # if you want to save all data
        pickle.dump(cifarMaxTransform, out)
        pickle.dump(cifarMinTransform, out)
        pickle.dump(cifarMeanTransform, out)
        pickle.dump(cstd, out)
    return cifarTransforms


# find std. deviation of each transform coefficient
def getStdDev(transformArray):
    out = np.zeros((32, 32, 3), dtype='float32')
    for row in range(32):
        for col in range(32):
            for ch in range(3):
                out[row, col, ch] = transformArray[:, row, col, ch].std()
    return out


# import RGB CIFAR100 batch file of 50k images
# google cifar for this and cifar10 dataset
def importCifar100():
    # cifar-100 is 1 file
    with open('data/cifar_raw_data/cifar100.pkl', 'rb') as fo:
        u = pickle._Unpickler(fo)
        u.encoding = 'Latin1'
        cifar_data = u.load()

    cifar = cifar_data['data']
    print('Imported dataset: Cifar100.')
    print('Samples: {}, shape: {}, datarange: {} to {}.'.format(len(cifar),
                                                                cifar.shape,
                                                                cifar.min(),
                                                                cifar.max()))
    return cifar


# import RGB CIFAR10 batch files of 10k images
# return 3072 x n np.array in range 0-255
# NEEDS TO BE SCALED to +/- 1
def importCifar10(howmany=5):
    # cifar 10 data in batches 1-5
    cifar10 = np.zeros((10000*howmany, 3072), dtype='uint8')
    for i in range(1, howmany+1):
        name = ''.join(['data/cifar_raw_data/data_batch_', str(i)])
        with open(name, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'Latin1'
            cifar_data = u.load()
        cifar10[(i-1)*10000:i*10000] = cifar_data['data']
    print('Imported dataset: Cifar10')
    print('Samples: {}, shape: {}, datarange: {} to {}.'.format(len(cifar10),
                                                                cifar10.shape,
                                                                cifar10.min(),
                                                                cifar10.max()))
    return cifar10


# paths to various data files
tiny_images_path = "../tiny_images/tiny_images.bin"

img_count = 79302017


def load_tiny(n=0):
    '''load nth image from tiny_images_path'''
    with open(tiny_images_path, 'rb') as f:
        # imgs = np.empty((n, 32, 32, 3))
        f.seek(3072*n)
        return np.fromstring(f.read(3072), dtype='uint8')\
                 .reshape((32, 32, 3), order='F')
