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


def get_test_img():
    test_images = []
    test_labels = []
    test_images.append('IMG/center_2016_12_01_13_46_24_718.jpg')
    test_images.append('IMG/center_2016_12_01_13_37_42_579.jpg')
    test_images.append('IMG/center_2016_12_01_13_38_06_234.jpg')
    test_labels.append(-0.2781274)
    test_labels.append(0.1765823)
    test_labels.append(-0.107229)

def preprocess(folder, fname):
    """ Creates a binary numpy format output of the steering images and
    angles"""
    data = pd.read_csv(folder + '//' + fname)
    zero_data = data[data.steering == 0] # take a smaller sample of more numerous zero angles
#    zero_data = zero_data.sample(int(0.05 * len(zero_data)))
    rest_data = data[~(data.steering == 0)]
    data = rest_data.append(zero_data)
    data = data.reindex(np.random.permutation(data.index))
    images = np.array([cv2.cvtColor(cv2.imread(folder + '//' + row.center),
                      cv2.COLOR_BGR2RGB) for index, row in data.iterrows()])
#                      cv2.COLOR_BGR2RGB) / 255.0 - 0.5 for index, row in data.iterrows()])
    labels = np.array([row.steering for index, row in data.iterrows()])
    np.save(folder + r'/images.npy', images)
    np.save(folder + r'/labels.npy', labels)
  
def add_bright(img):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    b_adder = .2 + np.random.uniform()
    img[:,:,2] = img[:,:,2] * b_adder
    img = cv2.cvtColor(img, cv2.COLORHSV2RGB)
    return img


def pick_image():
    check_val = np.random.random()
    if check_val < 0.33: # use left image
        offset = 0.25
        position = 'left'
    elif check_val < 0.67: # use center image
        offset = 0.0
        position = 'center'
    else:
        offset = -0.25
        position = 'right'
    return position, offset

    
# generator to get n images, n steering angle from input

def get_batch_data(folder, df, num, img_shape, augment=False, threshold=1.0):
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
    weights = np.empty(num,)
    while 1:
        np.random.seed()
        count = 0
        while count < num:
            next_idx = np.random.randint(0, len(df)-1)
            next_row = df.iloc[next_idx]
            if abs(next_row.steering) < 0.1:
                check_val = np.random.random()
                weight = 1
            else:
                check_val = 0.0
                weight = 10
            if check_val < threshold:
                if augment:
                    position, offset = pick_image()
                else:
                    offset = 0.0
                    position = 'center'
                labels[count] = next_row.steering + offset
                weights[count] = weight
                image = cv2.imread(folder + '//' + next_row[position].strip())
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if augment:
                    check_val = np.random.random()
                    if check_val < 0.5:
                        image = reverse_image(image)
                        labels[count] = -(next_row.steering + offset)
                    #image = add_bright(image)
                image = image[:-25, :, :]
                image = cv2.resize(image, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_AREA) 
                # resize takes width, height where hape is height, width
                images[count] = image
                count += 1
        #yield images, labels, weights
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
    #model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(1164))
    model.add(ELU())
    #model.add(Dropout(0.25))
    model.add(Dense(100))
    model.add(ELU())
    #model.add(Dropout(0.25))
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
    adm = Adam(lr=0.00001)
    log_csv = pd.read_csv(r'../simulator/data/driving_log.csv')
    model.compile(loss='mean_squared_error', optimizer=adm,
                  metrics=['accuracy', 'mean_squared_error'])
    checkpointer = ModelCheckpoint(filepath="checkpoint.h5", verbose=1, save_best_only=True)
    early_stop = EarlyStopping(monitor='val_mean_squared_error', min_delta=0.0001, patience=4,
                               verbose=1, mode='min')
    history = model.fit(train_features, train_labels, nb_epoch=n_epoch,
                        verbose=1, validation_split=0.2, batch_size=batch_size,
                        callbacks=[checkpointer, early_stop])
    return model, history


def train_nn_gen(model, batch_size, n_epoch):
    """ trains the keras model, uses generator for images, saves checkpoint data"""
    #adm = Adam(lr=0.00001)
    adm = Adam()
    log_csv = pd.read_csv(r'../simulator/data/driving_log.csv')
    data_gen = get_batch_data(r'../simulator/data', log_csv, batch_size, (100,
        200,3), augment=True)
    model.compile(loss='mean_squared_error', optimizer=adm,
                  metrics=['accuracy', 'mean_squared_error'])
    checkpointer = ModelCheckpoint(filepath="checkpoint.h5", verbose=1, save_best_only=True)
    early_stop = EarlyStopping(monitor='val_mean_squared_error', min_delta=0.0001, patience=4,
                               verbose=1, mode='min')
    model.fit_generator(data_gen, samples_per_epoch=24320, nb_epoch=n_epoch)
#            callbacks=[checkpointer, early_stop], validation_data=data_gen,
#            nb_val_samples=800)
    return model


def load_model(fname, model):
    model.compile(loss='mean_squared_error', optimizer='adam',
                  metrics=['accuracy', 'mean_squared_error'])
    weights = model.load_weights(fname)
    return model
    
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
    #input_shape = X_train.shape[1:]
    input_shape = (100, 200, 3)
    model = create_nn_nvidia(input_shape)
    print("Done creating model")
    model.summary()
    #model, history = train_nn(model, X_train, y_train, 128, 10) # batch size of 64, 5 epochs
    model = train_nn_gen(model, 128, 10)
    print("Done training model")
    export_nn(model)
    print("Done exporting model")
    test_img, test_label = get_test_img()
    for index, fname in enumerate(test_img):
        image = cv2.imread(folder + '//' + next_row[position].strip())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[:-25, :, :]
        image = cv2.resize(image, (200, 100), interpolation=cv2.INTER_AREA)
        result = model.predict(image)
        print('Result is {}, actual is {}'.format(result, test_label[index]))


if __name__ == '__main__':
    run_nn()
