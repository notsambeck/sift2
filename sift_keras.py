# load / train keras / tensorflow model
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Dropout, Dense, Activation
import pickle
import dataset
# import h5py # library is needed; import is not


model = Sequential()

model.add(Conv2D(32, (5, 5), padding='same', input_shape=(32, 32, 3),
                 data_format='channels_last'))
model.add(Activation('relu'))
model.add(Conv2D(64, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(units=1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(units=512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(2))
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

orig_data = 'data/keras_dataset_300k_{}.pkl'   # 0-9
import_batch_dset1 = 'data/import_batch_20170813_{}.pkl'  # 0-9

savefile = 'net/keras_net_v0_2017aug7.h5'
filelist1 = [import_batch_dset1.format(i) for i in range(9)]
filelist2 = [orig_data.format(i) for i in range(10)]
testfile = 'data/import_batch_20170813_9.pkl'  # 0-9

filelist = filelist1[:4] + filelist2[:5] + filelist1[4:] + filelist2[5:]
model.load_weights(savefile)


def train_net(load=savefile):
    '''train_net loads parameters from savefile by default,
    and then trains 100 epochs on multi-file dataset
    specified by filelist'''
    if load:
        try:
            model.load_weights(load)
        except:
            print('WARNING: failed to load weights but will overwrite file')
    else:
        print('warning: not saving progress.')
    print()
    for epoch in range(1000):
        for chunk in filelist1:
            with open(chunk, 'rb') as f:
                x = pickle.load(f)
                y = pickle.load(f)
                l = len(x)
            if l != len(y):
                raise ValueError('data/labels not equal length!')
            if len(y.shape) == 1:
                y = keras.utils.to_categorical(y, 2)

            model.fit(x, y, epochs=1, batch_size=1000)

        print('trained epoch {}; testing...'.format(epoch))
        with open(testfile, 'rb') as f:
            x = pickle.load(f)
            y = pickle.load(f)
            y = keras.utils.to_categorical(y, 2)
        e = model.evaluate(x, y, batch_size=2000)
        print(e)
        model.save(load)


def predict_and_show_incorrect(filename=testfile, limit=100):
    with open(filename, 'rb') as f:
        x = pickle.load(f)
        y = pickle.load(f)  # integer 0/1

    prs = model.predict(x)

    incorrect = 0
    for i in range(len(prs)):
        if prs[i, 0] > prs[i, 1]:   # predict 0
            p = 0
        else:
            p = 1

        if p != y[i]:
            incorrect += 1
            dataset.show_data(x, i=i)
            input('prediction = {}    press enter'.format(p))

        if incorrect == limit:
            return 'limit reached at i={}'.format(i)

    return 'of {} images, {} incorrect predictions'.format(len(prs),
                                                           incorrect)
