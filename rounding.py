import pickle
import numpy as np
import os


# Round all the values in the layer to X decimal places
def recurWeights(layer):
    if np.isscalar(layer[0]):
        x = 6
        for j in range(len(layer)):
            layer[j] = round(layer[j], x)
    else:
        for j in layer:
            recurWeights(j)


directory = r'./Networks/Original'

for filename in os.listdir(directory):
    f = open("./Networks/Original/%s"%filename, "rb")
    weights = pickle.load(f)
    weights[-1] = weights[-1].flatten()

    recurWeights(weights)

    pickle.dump(weights, open("./Networks/RoundingErrorSim/%s" % filename, "wb"))

