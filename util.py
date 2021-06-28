
import keras
import numpy as np

def get_weights(model):
    weights = []
    bias = []
    for layer in model.layers:
        arr = np.asarray(layer.get_weights())
        if len(arr) == 2:
            weights.append(arr[0])
            bias.append(arr[1])
    return weights,bias


def set_weights(model.weights,bias):
    weights = []
    bias = []
    counter = 0
    for layerInd in range(len(model.layers)):
        arr = np.asarray(model.layers[layerInd].get_weights())
        if len(arr) == 2:
            model.layers[layerInd].set_weights([weights[counter],bias[counter]])
            counter += 1
    return model
