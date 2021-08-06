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
    for i_d in range(len(model_weights)):
        old_shape = model_weights[i_d].shape
        model_weights[i_d] = model_weights[i_d].flatten()
        avg = np.average(model_weights[i_d])
        sd = np.std(model_weights[i_d])
        pdfs.append([avg, sd])
        # count, bins = np.histogram(model_weights[i_d], bins=100)
        # pdf = count / sum(count)
        # pdfs.append(pdf)
        # plt.plot(bins[0:100], pdf)
        # plt.title("Layer %d"%i_d)
        # plt.savefig("Layer %d Dist"%i_d)
        # plt.clf()

        model_weights[i_d] = model_weights[i_d].reshape(old_shape)

    pickle.dump(pdfs, open("./pdfs", "wb"))
    return model_weights

# model = keras.models.load_model("MNIST_model")
# weights, bias = get_weights(model)
#
# plot_layer(weights)




