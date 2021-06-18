import pickle
import numpy as np
import random
import os
import struct


def float_to_bin(num):
    return format(struct.unpack('!I', struct.pack('!f', num))[0], '032b')


def bin_to_float(binary):
    return struct.unpack('!f', struct.pack('!I', int(binary, 2)))[0]


# Flips one bit per layer
def recurWeights(layer):
    if np.isscalar(layer[0]):
        weightChoice = random.randrange(0, len(layer)-1)
        print(layer[weightChoice])
        change = float_to_bin(layer[weightChoice])
        shift = ""

        for i in range(32):
            if i == bitCount:
                shift += "1"
            else:
                shift += "0"
        print(shift)
        change = [str(int(change[i]) ^ int(shift[i])) for i in range(len(shift))]
        change = ''.join(change)
        layer[weightChoice] = bin_to_float(change)
        print(layer[weightChoice])
    else:
        for j in layer:
            recurWeights(j)


directory = r'./Networks/Original'

for filename in os.listdir(directory):
    for bitCount in range(32):
        f = open("./Networks/Original/%s" % filename, "rb")
        weights = pickle.load(f)
        weights[-1] = weights[-1].flatten()
        recurWeights(weights)

        pickle.dump(weights, open("./Networks/SingleBitCorrupted/Bit-%02d/%s" % (bitCount, filename), "wb"))

