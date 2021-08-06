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
    sample_size = 30
    for i_d in range(len(model_weights)):
        old_shape = model_weights[i_d].shape
        model_weights[i_d] = model_weights[i_d].flatten()

        healthy_avg = pdfs[i_d][0]
        healthy_std = pdfs[i_d][1]
        for i in range(0, len(model_weights[i_d])-sample_size, sample_size):
            local_avg = np.mean(model_weights[i_d][i:i+sample_size])
            if local_avg < healthy_avg-(2*healthy_std) or local_avg > healthy_avg+(2*healthy_std):
                error += 1
        model_weights[i_d] = model_weights[i_d].reshape(old_shape)

    return error


# model = keras.models.load_model("MNIST_model")
# weights, bias = get_weights(model)
#
# errors = check_layer(weights)
