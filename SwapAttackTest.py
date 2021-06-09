import pickle
import numpy as np
import os


def recurWeights(layer):
    if np.isscalar(layer[0]):
        temp = layer[0]
        layer[0] = layer[-1]
        layer[-1] = temp
    else:
        for j in layer:
            recurWeights(j)


directory = r'./Networks'

for filename in os.listdir(directory):
    f = open("./Networks/%s"%filename, "rb")
    weights = pickle.load(f)
    weights[-1] = weights[-1].flatten()

    recurWeights(weights)

    pickle.dump(weights, open("./FirstLastSwappedNetworks/%s" % filename, "wb"))
