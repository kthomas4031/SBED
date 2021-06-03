import pickle
import numpy as np
import random

f = open("./Networks/Model_1_elu_weights.pickle", "rb")
weights = pickle.load(f)

def random_corrupt(layer):
    for i in range(len(layer)):
        chance = random.randrange(0,19)
        if chance <= 1:
            layer[i] = random.randrange(-2, 2)

def recurWeights(layer):
    if np.isscalar(layer[0]):
        random_corrupt(layer)
    else:
        for j in layer:
            recurWeights(j)


recurWeights(weights)

pickle.dump(weights, open("./RandomlyCorruptedNetworks/Model_1_elu_weights.pickle", "wb"))
