import pickle
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dropout, MaxPooling2D, Convolution2D, Lambda, ELU
from keras.layers.core import Dense, Activation, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import model_from_json
from keras.optimizers import Adam
import pandas as pd
import tensorflow as tf
import cv2
tf.python.control_flow_ops = tf

log_csv = pd.read_csv(r'../simulator/data/driving_log.csv')


def shift_image(img, angle, max_shift):
    """ shift image by +/- max_shift in x, 0 in y, and adjust steering angle"""
    img_shape = img.shape
    x_shift = np.random.uniform(-max_shift / 2.0, max_shift / 2.0)
    shift_angle = angle + x_shift / max_shift * .4
    y_shift = 0
    shift_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    shift_img = cv2.warpAffine(img, shift_matrix, (img_shape[1],
            img_shape[0]))
    return shift_img, shift_angle


def get_test_img():
    """ 3 test images to see how well the model predicts steering angles"""
    test_images = []
    test_labels = []
    test_images.append('IMG/center_2016_12_01_13_46_24_718.jpg')
    test_images.append('IMG/center_2016_12_01_13_37_42_579.jpg')
    test_images.append('IMG/center_2016_12_01_13_38_06_234.jpg')
    test_labels.append(-0.2781274)
    test_labels.append(0.1765823)
    test_labels.append(-0.107229)
    return test_images, test_labels


def add_bright(img):
    """ modify brightness of image by 0.3 - 1.3"""
    img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    b_adder = .3 + np.random.uniform()
    img[:,:,2] = img[:,:,2] * b_adder
    #img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img


def pick_image(offset=0.25):
    """ return left, centre or right image with equal distribution"""
    check_val = np.random.randint(3)
    if check_val == 0: # use left image
        position = 'left'
    elif check_val == 1: # use center image
        offset = 0.0
        position = 'center'
    else: # use right image
        offset = -offset
        position = 'right'
    return position, offset

    
def get_batch_data(folder, df, num, img_shape, augment=False, threshold=1.0, offset=0.25):
    """ generator to return a small batch of images, labels
    folder: location of simulator data
    df: DataFrame of driving_log
    num: number of images in batch
    img_shape: desired image shape (row, col, c)
    augment: add data augmentation
    threshold: used to determine how many small steering angle images to include
    """
    images = np.empty([num, img_shape[0], img_shape[1], img_shape[2]])
    labels = np.empty(num,)
    while 1:
        np.random.seed()
        count = 0
        while count < num:
            next_idx = np.random.randint(len(df))
            next_row = df.iloc[next_idx]
            # exclude small angles, based on threshold
            if abs(next_row.steering) < 0.1:
                check_val = np.random.random()
            else:
                check_val = -1
            if check_val < threshold:
                if augment:
                    # choose between left, right, center images
                    position, offset = pick_image(offset=offset)
                else:
                    offset = 0.0
                    position = 'center'
                image = cv2.imread(folder + '//' + next_row[position].strip())
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                angle = next_row.steering + offset
                max_shift = 80
                # shift image, modify brightness, flip 50% of images
                if augment:
                    image, angle = shift_image(image, next_row.steering + offset, max_shift) 
                    check_val = np.random.random()
                    if check_val < 0.5: # 50% of the time, return flipped image
                        image = reverse_image(image)
                        angle = -angle
                    image = add_bright(image)
                labels[count] = angle
                image = image[40:-25, max_shift:-max_shift, :] # trim off car hood and top of image
                image = cv2.resize(image, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_AREA) 
                images[count] = image
                count += 1
        yield images, labels

def reverse_image(img):
    img = img[:, ::-1, :]
    return img


def create_nn_nvidia(input_shape):
    """ NN based on Nvidia model"""
    # check out https://arxiv.org/pdf/1604.07316v1.pdf
    # https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
    model = Sequential()
    model.add(Lambda(lambda x: x/255 -0.5, input_shape=input_shape)) #normalize image
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
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(1164))
    model.add(ELU())
    model.add(Dropout(0.2))
    model.add(Dense(100))
    model.add(ELU())
    model.add(Dropout(0.2))
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
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=input_shape))
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
    """ trains the keras model with preloaded data, saves checkpoint data"""
    adm = Adam(lr=0.00001)
    model.compile(loss='mean_squared_error', optimizer=adm,
                  metrics=['accuracy', 'mean_squared_error'])
    checkpointer = ModelCheckpoint(filepath="checkpoint.h5", verbose=1, save_best_only=True)
    early_stop = EarlyStopping(monitor='val_mean_squared_error', min_delta=0.0001, patience=4,
                               verbose=1, mode='min')
    history = model.fit(train_features, train_labels, nb_epoch=n_epoch,
                        verbose=1, validation_split=0.2, batch_size=batch_size,
                        callbacks=[checkpointer, early_stop])
    return model, history


def train_nn_gen(model, batch_size, n_epoch, img_shape, per_epoch, offset=0.25, lr=0.001,
        threshold = 1.0):
    """ trains the keras model, uses generator for images, saves checkpoint data"""
    #adm = Adam(lr=0.00001)
    adm = Adam(lr=lr)
    data_gen = get_batch_data(r'../simulator/data', log_csv, batch_size, img_shape, 
                              augment=True, offset=offset, threshold=threshold)
    val_data_gen = get_batch_data(r'../simulator/data', log_csv, batch_size, img_shape, 
                              augment=False, offset=offset, threshold=threshold)
    model.compile(loss='mean_squared_error', optimizer=adm,
                  metrics=['accuracy', 'mean_squared_error'])
    checkpointer = ModelCheckpoint(filepath="checkpoint.h5", verbose=1, save_best_only=True)
    early_stop = EarlyStopping(monitor='val_mean_squared_error', min_delta=0.0001, patience=4,
                               verbose=1, mode='min')
    model.fit_generator(data_gen, samples_per_epoch=per_epoch, nb_epoch=n_epoch,
            callbacks=[checkpointer, early_stop], validation_data=val_data_gen,
            nb_val_samples=math.floor(per_epoch / 8))
    return model


def load_model(fname):
    json_handle = open(fname + '.json', 'r')
    model = model_from_json(json_handle.read())
    model.load_weights(fname + '.h5')
    return model
    
def export_nn(model, fname):
    """ saves the model and weights"""
    model_json = model.to_json()
    with open(fname + ".json", 'w') as model_file:
        model_file.write(model_json)
    model.save_weights(fname + '.h5')


def explore_nn():
    """ create models to evaluate different parameter values """
    input_shape = (64, 64, 3)
    batch_size = 256
    samples_per_epoch = 48*1024
    count = 0
    model = create_nn_nvidia(input_shape)
    model.summary()
    print("Done creating model")
    print("final run, 10 epochs with increased low angle threshold/reduced learning rate")
    while count < 10:
        if count < 8:
            current_threshold = 0.8
            current_lr = 0.001
        else:
            current_threshold = 1.0
            current_lr = 0.0001
            
        print('Evaluating threshold = {}, lr = {}'.format(current_threshold,
                                                          current_lr))
        model = train_nn_gen(model, batch_size, 1, input_shape,
                samples_per_epoch, threshold=current_threshold, lr=current_lr)
        print("Done training model")
        export_nn(model, 'model' + str(count))
        print("Done exporting interim model")
        count += 1
    export_nn(model, 'model')
    print("Done exporting final model")


def run_nn():
    input_shape = (64, 64, 3)
    model = create_nn_nvidia(input_shape)
    print("Done creating model")
    model.summary()
    batch_size = 256
    epochs = 4
    model = train_nn_gen(model, batch_size, epochs, input_shape, 32768,
            threshold=0.8)
    print("Done training model")
    export_nn(model, 'model')
    print("Done exporting model")
    test_img, test_label = get_test_img()
    for index, fname in enumerate(test_img):
        image = cv2.imread(r'../simulator/data' + '//' + fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[40:-25, 80:-80, :]
        image = cv2.resize(image, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        result = model.predict(image[None, :, :, :], batch_size=1)
        print('Result is {}, actual is {}'.format(result, test_label[index]))


if __name__ == '__main__':
    #run_nn()
    explore_nn()
