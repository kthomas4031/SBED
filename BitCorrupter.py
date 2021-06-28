import pickle
import numpy as np
import random
import os
import struct


def float_to_bin(num):
    return format(struct.unpack('!I', struct.pack('!f', num))[0], '032b')


def bin_to_float(binary):
    return struct.unpack('!f', struct.pack('!I', int(binary, 2)))[0]


def flipBit(layer):
    weightChoice = random.randrange(0, len(layer) - 1)
    change = float_to_bin(layer[weightChoice])
    shift = ""

    for i in range(32):
        if i == bitFlipped:
            shift += "1"
        else:
            shift += "0"

    change = [str(int(change[i]) ^ int(shift[i])) for i in range(len(shift))]
    change = ''.join(change)

    layer[weightChoice] = bin_to_float(change)
    return layer


# Flips one bit per layer
def recurWeights(layer):
    if np.isscalar(layer[0]):
        return flipBit(layer)

    else:
        for j in range(len(layer)):
            layer[j] = recurWeights(layer[j])
        return layer


directory = r'./Networks/Original'


for filename in os.listdir(directory):
    f = open("./Networks/Original/%s" % filename, "rb")
    weights = pickle.load(f)
    weights[-1] = weights[-1].flatten()

    for bitFlipped in range(32):
        print("Bit: %d" %bitFlipped)
        print("===================================")
        print(weights)
        changedWeights = recurWeights(weights)
        print("-----------------------------------")
        print(changedWeights)
        print("===================================")
        pickle.dump(changedWeights, open("./Networks/SingleBitCorrupted/Bit-%02d/%s" % (bitFlipped, filename), "wb"))

    f.close()