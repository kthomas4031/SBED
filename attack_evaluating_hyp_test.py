# first neural network with keras make predictions
import keras
import numpy as np
from tensorflow.keras.datasets import cifar10, mnist
from tensorflow.keras.utils import to_categorical
import struct
import random
from hypothesis_detector import *
from hypothesis_detection_init import *
import time
import sys


def float_to_bin(num):
    return format(struct.unpack('!I', struct.pack('!f', num))[0], '032b')


def bin_to_float(binary):
    return struct.unpack('!f', struct.pack('!I', int(binary, 2)))[0]


def manipulate_cipher_bit(model_weights, per_weight_manipulation):
    # for i_d in range(len(model_weights)):
    i_d = random.randrange(0, len(model_weights)-1)
    old_shape = model_weights[i_d].shape
    model_weights[i_d] = model_weights[i_d].flatten()
    for j in range(per_weight_manipulation):
        rand = random.randrange(0, len(model_weights[i_d]) - 8)
        change = [float_to_bin(model_weights[i_d][rand]), float_to_bin(model_weights[i_d][rand + 1]),
                  float_to_bin(model_weights[i_d][rand + 2]), float_to_bin(model_weights[i_d][rand + 3]),
                  float_to_bin(model_weights[i_d][rand + 4]), float_to_bin(model_weights[i_d][rand + 5]),
                  float_to_bin(model_weights[i_d][rand + 6]), float_to_bin(model_weights[i_d][rand + 7])]
        for i_s in range(8):
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


def manipulate_single_bit(model_weights, per_weight_manipulation, bit_position):
    # for i_d in range(len(model_weights)):
    i_d = random.randrange(0, len(model_weights) - 1)
    old_shape = model_weights[i_d].shape
    model_weights[i_d] = model_weights[i_d].flatten()
    for j in range(per_weight_manipulation):
        rand = random.randrange(0, len(model_weights[i_d]) - 1)
        change = float_to_bin(model_weights[i_d][rand])
        shift = ""
        # rand_mod = random.randint(0, 31)
        rand_mod = bit_position
        # rand_mod = 10
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


def zero_out(model_weights):
    for i_d in range(len(model_weights)):
        old_shape = model_weights[i_d].shape
        model_weights[i_d] = model_weights[i_d].flatten()
        for j in range(len(model_weights[i_d])):
            model_weights[i_d][j] = -20
        model_weights[i_d] = model_weights[i_d].reshape(old_shape)
    return model_weights


def manipulate_permutation(model_weights):
    rand = random.randint(0, len(model_weights) - 1)
    rand2 = random.randint(0, len(model_weights[rand])-1)
    old_shape = model_weights[rand][rand2].shape
    model_weights[rand][rand2] = model_weights[rand][rand2].flatten()
    model_weights[rand][rand2] = np.random.permutation(model_weights[rand][rand2])
    model_weights[rand][rand2] = model_weights[rand][rand2].reshape(old_shape)
    return model_weights


def get_weights(model):
    weights = []
    bias = []
    for layer in model.layers:
        arr = np.asarray(layer.get_weights())
        if len(arr) == 2:
            weights.append(arr[0])
            bias.append(arr[1])
    return weights, bias


def set_weights(model, weights, bias):
    counter = 0
    for layerInd in range(len(model.layers)):
        arr = np.asarray(model.layers[layerInd].get_weights())
        if len(arr) == 2:
            model.layers[layerInd].set_weights([weights[counter], bias[counter]])
            counter += 1
    return model

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

bit_pos = 0

try:
    # for bit_pos in range(2,32):
    errorsDetected = 0
    acc_avg = []
    t_avg = []
    loss_avg = []
    big_weights = []
    for i in range(1000):
        model = keras.models.load_model("MNIST_model")
        weights, bias = get_weights(model)

        # Initialize for error detection
        plot_layer(weights)

        bit_flipped_weights = manipulate_cipher_bit(weights, 1)

        bit_flipped_model = set_weights(model, bit_flipped_weights, bias)

        score = bit_flipped_model.evaluate(x_test, y_test, verbose=0)
        loss_avg.append(score[0])
        acc_avg.append(score[1])

        # Detect Error
        t0 = time.time()
        errors = check_layer(bit_flipped_weights)
        if errors > 0:
            errorsDetected += errors
        t1 = time.time()
        t_avg.append(t1 - t0)

        bit_flipped_weights = np.absolute(bit_flipped_weights)
        for i in range(len(bit_flipped_weights)):
            bit_flipped_weights[i] = bit_flipped_weights[i].flatten()

        for i in range(len(bit_flipped_weights)):
            for j in range(len(bit_flipped_weights[i])):
                if(bit_flipped_weights[i][j] > 1.05):
                    big_weights.append(bit_flipped_weights[i][j])


    acc_avg = sum(acc_avg) / len(acc_avg)
    loss_avg = sum(loss_avg) / len(loss_avg)
    big_weights = [abs(ele) for ele in big_weights]
    big_weights = sorted(big_weights)
    if len(big_weights) > 0:
        perc_of_errors = errorsDetected/len(big_weights)
    else:
        perc_of_errors = 0
    print("Bit Pos: %s\nAvg Acc: %f\nAvg Loss: %f\nTime for Check: %f\nErrors Detected: %f\nErrors:%d\nPercent Errors: %f"
          %(bit_pos, acc_avg, loss_avg, t_avg[0], errorsDetected, len(big_weights), perc_of_errors))
    print(big_weights)

except:
    print("Exception Occurred")

    acc_avg = sum(acc_avg)/len(acc_avg)
    loss_avg = sum(loss_avg)/len(loss_avg)
    print("Avg Acc: %f\nAvg Loss: %f\nTime for Check: %f\nErrors Detected: %f"
          %(acc_avg, loss_avg, t_avg[0], errorsDetected))
