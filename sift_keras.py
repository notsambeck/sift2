# train model; this will go to train.py
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Dropout, Dense, Activation
import pickle
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

opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


savefile = 'net/keras_net_v0_2017aug7.h5'
filelist = ['data/keras_dataset_300k_{}.pkl'.format(i) for i in range(9)]
testfile = 'data/keras_dataset_300k_9.pkl'


def train_net():
    for epoch in range(100):
        for chunk in filelist:
            with open(chunk, 'rb') as f:
                x = pickle.load(f)
                y = pickle.load(f)
                l = len(x)
            if l != len(y):
                raise ValueError('data/labels not equal length!')
            if len(y.shape) == 1:
                y = keras.utils.to_categorical(y, 2)

            model.fit(x, y, epochs=1, batch_size=2000)

        print('trained epoch {}; testing...'.format(epoch))
        with open(testfile, 'rb') as f:
            x = pickle.load(f)
            y = pickle.load(f)
            y = keras.utils.to_categorical(y, 2)
        e = model.evaluate(x, y, batch_size=2000)
        print(e)
        model.save(savefile)
