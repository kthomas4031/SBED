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
        weightChoice = random.randrange(0, len(layer))
        #layer[weightChoice] = bitflip(layer[weightChoice], random.randrange(0, 6))
        layer[weightChoice] += 2
    else:
        for j in layer:
            recurWeights(j)


directory = r'./Networks'

for filename in os.listdir(directory):
    f = open("./Networks/%s"%filename, "rb")
    weights = pickle.load(f)
    weights[-1] = weights[-1].flatten()

    recurWeights(weights)

    pickle.dump(weights, open("./SingleBitCorruptedNetworks/%s" % filename, "wb"))

