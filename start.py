import pickle
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dropout, MaxPooling2D, Convolution2D, Lambda, ELU
from keras.layers.core import Dense, Activation, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import model_from_json
import pandas as pd
import tensorflow as tf
import cv2
tf.python.control_flow_ops = tf


def preprocess(folder, fname):
    """ Creates a binary numpy format output of the steering images and
    angles"""
    data = pd.read_csv(folder + '//' + fname)
    zero_data = data[data.steering == 0] # take a smaller sample of more numerous zero angles
    zero_data = zero_data.sample(int(0.05 * len(zero_data)))
    rest_data = data[~(data.steering == 0)]
    data = rest_data.append(zero_data)
    data = data.reindex(np.random.permutation(data.index))
    images = np.array([cv2.imread(folder + '//' + row.center) / 255.0 - 0.5 for index, row in data.iterrows()])
    labels = np.array([row.steering for index, row in data.iterrows()])
    np.save(folder + r'/images.npy', images)
    np.save(folder + r'/labels.npy', labels)
  

# generator to get n images, n steering angle from input
def get_data(fname, num):
    """ generator to return a small batch of images, labels"""
    current_row = 0
    source = pd.read_csv(fname)
    while 1:
        temp = source.iloc[current_row: current_row + num]
        current_row += num
        images = get_images(temp) # need to add get_images
        labels = np.asarray(temp['steering'])
        yield (images, labels)


def create_nn_nvidia(input_shape):
    """ NN based on Nvidia model"""
    # check out https://arxiv.org/pdf/1604.07316v1.pdf
    # https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
    model = Sequential()
    model.add(Convolution2D(24, 5, 5,
                            border_mode='valid',
                            subsample=(2, 2),
                            input_shape=input_shape))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5,
                            border_mode='valid',
                            subsample=(2, 2)))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5,
                            border_mode='valid',
                            subsample=(2, 2)))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3,
                            border_mode='valid',
                            subsample=(1, 1)))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3,
                            border_mode='valid',
                            subsample=(1, 1)))
    model.add(ELU())
#    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(1164))
    model.add(ELU())
    model.add(Dense(100))
    model.add(ELU())
    model.add(Dense(50))
    model.add(ELU())
    model.add(Dense(10))
    model.add(ELU())


    # final layer - output is a single digit (steering angle)
    model.add(Dense(1))

    return model


def create_nn_comma(input_shape):
    """ create a keras model based on comma.ai
    https://github.com/commaai/research"""
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 -1., input_shape=input_shape))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    return model


def create_nn(input_shape):
    """ create a basic keras model for pipecleaning, based on LeNet"""
    model = Sequential()
    # 1st Layer - Add a conv layer
    model.add(Convolution2D(32, 3, 3,
                            border_mode='valid',
                            input_shape=input_shape))
    # 2nd Layer - Add a pool layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))


    # 3rd Layer - Add a flatten layer
    model.add(Flatten())

    # 4th Layer - Add a fully connected layer
    model.add(Dense(128))

    # 5th Layer - Add a ReLU activation layer
    model.add(Activation('relu'))

    # final layer - output is a single digit (steering angle)
    model.add(Dense(1))

    return model

def train_nn(model, train_features, train_labels, batch_size, n_epoch):
    """ trains the keras model, saves checkpoint data"""
    model.compile(loss='mean_squared_error', optimizer='adam',
                  metrics=['accuracy', 'mean_squared_error'])
    checkpointer = ModelCheckpoint(filepath="checkpoint.h5", verbose=1, save_best_only=True)
    early_stop = EarlyStopping(monitor='val_mean_squared_error', min_delta=0.0001, patience=4,
                               verbose=1, mode='min')
    history = model.fit(train_features, train_labels, nb_epoch=n_epoch,
                        verbose=1, validation_split=0.2, batch_size=batch_size,
                        callbacks=[checkpointer, early_stop])
    return model, history


def export_nn(model):
    """ saves the model and weights"""
    model_json = model.to_json()
    with open("model.json", 'w') as model_file:
        model_file.write(model_json)
    model.save_weights('model.h5')


def run_nn():
    # load previously prepared data
    X_train = np.load("..//simulator//data//images.npy")
    y_train = np.load("..//simulator//data//labels.npy")
    print("Done loading data")
    input_shape = X_train.shape[1:]
    model = create_nn_nvidia(input_shape)
    print("Done creating model")
    model.summary()
    model, history = train_nn(model, X_train, y_train, 128, 3) # batch size of 64, 5 epochs
    print("Done training model")
    export_nn(model)
    print("Done exporting model")
    out = model.predictions(X_train)
    print(np.histogram, out)


if __name__ == '__main__':
    run_nn()
