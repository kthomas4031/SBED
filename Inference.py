
import pickle
import numpy as np
import matplotlib.pyplot as plt

f = open("./RandomlyCorruptedNetworks/Model_1_elu_weights.pickle", "rb")
weights = pickle.load(f)
weights = np.asarray(weights, dtype=object)
weights[-1] = weights[-1].flatten()

# f = open("./modelStats.pickle", "rb")
# storedStats = pickle.load(f)

g = open("./elu/1layerpdfs.pickle", "rb")
pdfs = pickle.load(g)
h = open("./elu/1layercdfs.pickle", "rb")
cdfs = pickle.load(h)
b = open("./elu/1bins.pickle", "rb")
bins_count = pickle.load(b)

errorLayers = []
layerNum = 0

def testDists(layer, numLayer):
    # Getting data of the histogram
    count, bins = np.histogram(layer, bins=10)
    error = 0
    # Finding the PDF of the histogram using count values
    layerpdf = count / sum(count)

    # Calculate the CDF
    layercdf = np.cumsum(layerpdf)

    for i in range(len(layerpdf)):
        if layerpdf[i] != pdfs[numLayer][i] or layercdf[i] != cdfs[numLayer][i]:
            error = 1

    if error == 1:
        print("Layer %d Error" % numLayer)
        errorLayers.append(numLayer)

# def testStats(layer, count):
#     # Initialize stats for current layer to ensure consistency
#     denom = 0
#     avg = 0
#
#     # Iterate through parameters in the layer and calculate stats
#     for parameter in layer:
#         denom += 1
#         avg += parameter
#
#         # Test min-max
#         if parameter > storedStats[count][3]:
#             #avg -= parameter
#             # Pull Replacement Weight
#             print("ERROR: EXCEEDED MAX AT " + str(count))
#             errorLayers.append(count)
#             #avg += parameter
#         elif parameter < storedStats[count][2]:
#             #avg -= parameter
#             # Pull Replacement Weight
#             print("ERROR: EXCEEDED MIN AT " + str(count))
#             errorLayers.append(count)
#             #avg += parameter
#
#     # Test that average is within boundary
#     avg = avg / denom
#     if abs(avg) > abs((storedStats[count][1]*1.1)) or abs(avg) < abs((storedStats[count][1]*0.9)):
#         # Pull Replacement Layer
#         print("ERROR: INVALID AVG AT " + str(count))
#         print(str(avg) + " vs " + str(storedStats[count][1]))
#         errorLayers.append(count)

# TESTING
# for i in range(0,20):
#     plt.plot(bins_count[i][1:], pdfs[i], color="red", label="PDF%i" % i)
#     plt.plot(bins_count[i][1:], cdfs[i], label="CDF%i" % i)
#     plt.legend()
#
# plt.show()

#Iterate through each layer

def recurWeights(layer):
    global layerNum
    if np.isscalar(layer[0]):
        #testStats(layer, layerNum)
        testDists(layer, layerNum)
        layerNum += 1
    else:
        for j in layer:
            recurWeights(j)


recurWeights(weights)


