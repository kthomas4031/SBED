from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import keras
import numpy as np
import pickle
from keras.models import model_from_json
from tensorflow.keras.datasets import cifar10, mnist
from tensorflow.keras.utils import to_categorical
from keras import layers
import sys
import struct
import random


# Model / data parameters
# num_classes = 10
# input_shape = (28, 28, 1)
#
# # the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# y_train = to_categorical(y_train, num_classes)
# y_test = to_categorical(y_test, num_classes)
#
# # Scale images to the [0, 1] range
# x_train = x_train.astype("float32") / 255
# x_test = x_test.astype("float32") / 255
# # Make sure images have shape (28, 28, 1)
# x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)
# print("x_train shape:", x_train.shape)
# print(x_train.shape[0], "train samples")
# print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
# model = keras.Sequential(
#     [
#         layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Flatten(),
#         layers.Dropout(0.5),
#         layers.Dense(num_classes, activation="softmax"),
#     ]
# )
# batch_size = 64
# epochs = 20
#
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# model = Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.2))
# model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.3))
# model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.4))
# model.add(layers.Flatten())
# model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
# model.add(layers.BatchNormalization())
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(10, activation='softmax'))
# compile model
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#
# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

def float_to_bin(num):
    return format(struct.unpack('!I', struct.pack('!f', num))[0], '032b')


def bin_to_float(binary):
    return struct.unpack('!f', struct.pack('!I', int(binary, 2)))[0]


def get_weights(model):
    weights = []
    bias = []
    for layer in model.layers:
        arr = np.asarray(layer.get_weights())
        if len(arr) == 2:
            weights.append(arr[0])
            bias.append(arr[1])
    return weights, bias


def manipulate_cipher_bit(model_weights, per_weight_manipulation):
    # for i_d in range(len(model_weights)):
    i_d = random.randrange(0, len(model_weights)-1)
    old_shape = model_weights[i_d].shape
    model_weights[i_d] = model_weights[i_d].flatten()
    for j in range(per_weight_manipulation):
        rand = random.randrange(0, len(model_weights[i_d]) - 4)
        change = [float_to_bin(model_weights[i_d][rand]), float_to_bin(model_weights[i_d][rand+1]),
                  float_to_bin(model_weights[i_d][rand+2]), float_to_bin(model_weights[i_d][rand+3])]
        for i_s in range(4):
            shift = ""
            rand_mod = random.sample(range(31), 16)
            for i in range(32):
                if i in rand_mod:
                    shift += "1"
                else:
                    shift += "0"
            change[i_s] = [str(int(change[i_s][i]) ^ int(shift[i])) for i in range(32)]
            change[i_s] = ''.join(change[i_s])
            model_weights[i_d][rand+i_s] = bin_to_float(change[i_s])
    model_weights[i_d] = model_weights[i_d].reshape(old_shape)
    return model_weights


def manipulate_single_bit(model_weights, per_weight_manipulation):
    i_d = random.randrange(0, len(model_weights) - 1)
    old_shape = model_weights[i_d].shape
    model_weights[i_d] = model_weights[i_d].flatten()
    for j in range(per_weight_manipulation):
        rand = random.randrange(0, len(model_weights[i_d]) - 1)
        change = float_to_bin(model_weights[i_d][rand])
        shift = ""
        # rand_mod = random.randint(0, 31)
        # rand_mod = bit_position
        rand_mod = 1
        for i in range(32):
            if i == rand_mod:
                shift += "1"
            else:
                shift += "0"
        change = [str(int(change[i]) ^ int(shift[i])) for i in range(32)]
        change = ''.join(change)
        model_weights[i_d][rand] = bin_to_float(change)
    model_weights[i_d] = model_weights[i_d].reshape(old_shape)
    return model_weights

means = []
sigmas = []

for ix in range(100):
    model = keras.models.load_model("MNIST_model")
    weight, bias = get_weights(model)

    weighty = manipulate_cipher_bit(weight, 1)

    weighty = np.asarray(weighty)

    for i in range(len(weighty)):
        weighty[i] = weighty[i].flatten()

    weights = []

    for i in range(len(weighty)):
        for j in range(len(weighty[i])):
            weights.append(weighty[i][j])

    mean = np.mean(weights)
    sigma = np.std(weights, dtype=np.float32)

    means.append(mean)
    sigmas.append(sigma)

    print(ix, "\nMean:  ", mean, "\nSigma: ", sigma)

means = np.asarray(means)
sigmas = np.asarray(sigmas)

min_mean = means[abs(means).argmin()]
min_sigma = sigmas[np.isclose(means, min_mean)]
max_mean = means[abs(means).argmax()]
max_sigma = sigmas[np.isclose(means, max_mean)]

avg_mean = np.mean(means)
avg_sigma = np.mean(sigmas)

print("Minimum Mean and Sigma = ", min_mean, ", ", min_sigma, "\nMaximum Mean and Sigma= ", max_mean, ", ", max_sigma,
      "\nAverage Mean and Sigma = ", avg_mean, ", ", avg_sigma, "\n")
