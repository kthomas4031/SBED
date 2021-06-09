import pickle
import numpy as np
import random
import os


def random_corrupt(layer):
    for i in range(len(layer)):
        chance = random.uniform(0, 1999)
        if chance <= 1:
            layer[i] += random.uniform(-1.5, 1.5)


def recurWeights(layer):
    if np.isscalar(layer[0]):
        random_corrupt(layer)
    else:
        for j in layer:
            recurWeights(j)


directory = r'./Networks/Original'

for filename in os.listdir(directory):
    f = open("./Networks/Original/%s"%filename, "rb")
    weights = pickle.load(f)
    weights[-1] = weights[-1].flatten()

    recurWeights(weights)

    pickle.dump(weights, open("./Networks/RandomlyCorrupted/%s" % filename, "wb"))
