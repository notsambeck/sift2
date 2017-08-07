# train model; this will go to train.py
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Dropout, Dense, Activation
import pickle

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3),
                 data_format='channels_last'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(units=1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(units=512))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


savefile = 'net/keras_net_v0_2017aug7.nn'
filelist = ['data/keras_dataset_300k_{}.pkl'.format(i) for i in range(9)]
testfile = 'data/keras_dataset_300k_9.pkl'
for epoch in range(100):
    if epoch % 10 == 9:
        print('Epoch {}; saving model to: {}'.format(epoch, savefile))
        model.save(savefile)
    for chunk in filelist:
        pr
        with open(chunk, 'rb') as f:
            x = pickle.load(f)
            y = pickle.load(f)
        l = len(x)
        if l != len(y):
            raise ValueError('data/labels not equal length!')
        if len(y.shape) == 1:
            y = keras.utils.to_categorical(y, 2)

        model.train_on_batch(x, y)  # this should return loss

    print('trained; testing')
    with open(testfile) as f:
        x = pickle.load(f)
        y = pickle.load(f)
    y_pred = model.predict(x)
    correct = sum([1 for i in range(len(y)) if y[i] == y_pred[i]])
    print('predicted {} out of {}. saving...'.format(correct, len(y)))
    model.save(savefile)
