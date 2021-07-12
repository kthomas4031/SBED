import pickle
import numpy as np
import os


def swapAttack(layer):
    if np.isscalar(layer[0]):
        temp = layer[0]
        layer[0] = layer[-1]
        layer[-1] = temp
        return layer
    else:
        for j in range(len(layer)):
            layer[j] = swapAttack(layer[j])
        return layer


# directory = r'./Networks'
#
# for filename in os.listdir(directory):
#     f = open("./Networks/%s"%filename, "rb")
#     weights = pickle.load(f)
#     weights[-1] = weights[-1].flatten()
#
#     weights = swapAttack(weights)
#
#     pickle.dump(weights, open("./FirstLastSwappedNetworks/%s" % filename, "wb"))
