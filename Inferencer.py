import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def checkdists(layer, numLayer, pdfs, cdfs):
    # Getting data of the histogram
    count, bins = np.histogram(layer, bins=100)

    # Finding the PDF of the histogram using count values
    layerpdf = count / sum(count)

    # Calculate the CDF
    layercdf = np.cumsum(layerpdf)
    sumPDF = sum(abs(pdfs[numLayer] - layerpdf))
    sumCDF = sum(abs(cdfs[numLayer] - layercdf))

    if sumCDF >= 0.015 or sumPDF >= 0.015:
        # print("Layer %d Error Diff = %f" % (numLayer, sumPDF))
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

#
# for i in range(0,20):
#     plt.plot(bins_count[i][1:], pdfs[i], color="red", label="PDF%i" % i)
#     plt.plot(bins_count[i][1:], cdfs[i], label="CDF%i" % i)
#     plt.legend()
#
# plt.show()

#Iterate through each layer


def recurWeights(layers, pdfs, cdfs):
    global layerNum
    if np.isscalar(layers[0]):
        checkdists(layers, layerNum, pdfs, cdfs)
        layerNum += 1
    else:
        for j in layers:
            recurWeights(j, pdfs, cdfs)


def inferenceCalc(network_weights):
    network_weights = np.asarray(network_weights, dtype=object)
    network_weights[-1] = network_weights[-1].flatten()
    g = open("./pdfsMNIST", "rb")
    pdfs = pickle.load(g)
    h = open("./cdfsMNIST", "rb")
    cdfs = pickle.load(h)
    recurWeights(network_weights, pdfs, cdfs)
    return errorLayers

# filename = "Model_1_elu_weights.pickle"
#
# for fold in os.listdir("./Networks/SingleBitCorrupted/"):
#     f = open("./Networks/SingleBitCorrupted/%s/%s" %(fold, filename), "rb")
#     weights = pickle.load(f)
#     weights = np.asarray(weights, dtype=object)
#     weights[-1] = weights[-1].flatten()
#
#
#     g = open("./elu/pdfs%s"%filename, "rb")
#     pdfs = pickle.load(g)
#     h = open("./elu/cdfs%s"%filename, "rb")
#     cdfs = pickle.load(h)
#     b = open("./elu/bins%s"%filename, "rb")
#     bins_count = pickle.load(b)
#
#     # totalNetwork = []
errorLayers = []
layerNum = 0
#
#     recurWeights(weights)

