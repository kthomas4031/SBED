import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10, mnist
from tensorflow.keras.utils import to_categorical
import keras

def get_weights(model):
    weights = []
    bias = []
    for layer in model.layers:
        arr = np.asarray(layer.get_weights())
        if len(arr) == 2:
            weights.append(arr[0])
            bias.append(arr[1])
    return weights, bias


def plot_layer(model_weights):
    pdfs = []
    for i_o in range(len(model_weights)):
        for i_d in range(len(model_weights[i_o])):
            layer = model_weights[i_o][i_d].flatten()

            avg = np.average(layer)
            sd = np.std(layer)
            pdfs.append([avg, sd])

    pickle.dump(pdfs, open("./pdfs", "wb"))
    return model_weights

# model = keras.models.load_model("MNIST_model")
# weights, bias = get_weights(model)
#
# plot_layer(weights)




