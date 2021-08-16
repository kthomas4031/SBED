from tensorflow.keras.datasets import cifar10, mnist
from tensorflow.keras.utils import to_categorical
import keras
import pickle
import numpy as np

def get_weights(model):
    weights = []
    bias = []
    for layer in model.layers:
        arr = np.asarray(layer.get_weights())
        if len(arr) == 2:
            weights.append(arr[0])
            bias.append(arr[1])
    return weights, bias


def check_layer(model_weights):
    g = open("./pdfs", "rb")
    pdfs = pickle.load(g)
    error = 0
    sample_size = 1
    loc = 0
    std_size = 3
    for i_o in range(len(model_weights)):
        for i_d in range(len(model_weights[i_o])):
            layer = model_weights[i_o][i_d].flatten()

            healthy_avg = pdfs[loc][0]
            healthy_std = pdfs[loc][1]
            for i in range(0, len(layer)-sample_size+1, sample_size):
                local_avg = layer[i] #np.mean(layer[i:i+sample_size])
                # if local_avg < healthy_avg-(std_size*healthy_std) or local_avg > healthy_avg+(std_size*healthy_std):
                if local_avg > 1.06 or local_avg < -1.06:
                    error += 1
            loc += 1
    return error

# 2.5 = 2.268
# 3 = 2.576
# model = keras.models.load_model("MNIST_model")
# weights, bias = get_weights(model)
#
# errors = check_layer(weights)
