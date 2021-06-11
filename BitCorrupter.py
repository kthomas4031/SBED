import pickle
import numpy as np
import random
import os
from struct import pack, unpack


def bitflip(x,pos):
    fs = pack('f',x)
    bval = list(unpack('BBBB',fs))
    [q,r] = divmod(pos,8)
    bval[q] ^= 1 << r
    fs = pack('BBBB', *bval)
    fnew=unpack('f',fs)
    return fnew[0]


# Flips one bit per layer
def recurWeights(layer):
    if np.isscalar(layer[0]):
        weightChoice = random.randrange(0, len(layer)-4)
        #layer[weightChoice] = bitflip(layer[weightChoice], random.randrange(0, 6))
        layer[weightChoice] += 0.1
        layer[weightChoice+1] += 0.1
        layer[weightChoice+1] += 0.1
        layer[weightChoice+1] += 0.1
    else:
        for j in layer:
            recurWeights(j)


directory = r'./Networks/Original'

for filename in os.listdir(directory):
    f = open("./Networks/Original/%s"%filename, "rb")
    weights = pickle.load(f)
    weights[-1] = weights[-1].flatten()

    recurWeights(weights)

    pickle.dump(weights, open("./Networks/CypherBitCorrupted/%s" % filename, "wb"))

