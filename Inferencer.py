import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def checkdists(layer, numLayer):
    # Getting data of the histogram
    count, bins = np.histogram(layer, bins=100)
    # Finding the PDF of the histogram using count values
    layerpdf = count / sum(count)

    # Calculate the CDF
    layercdf = np.cumsum(layerpdf)

    sumPDF = abs(pdfs[numLayer] - layerpdf)**2
    sumDiff = sum(sumPDF)/len(layerpdf)

    if sumDiff > 0:
        print("Layer %d Error Diff = %f" % (numLayer, sumDiff))
        errorLayers.append(numLayer)


    # plt.plot(bins[1:], layerpdf, color="blue", label="PDF%i" %numLayer)
    # plt.plot(bins_count[numLayer][1:], pdfs[numLayer], color="orange", label="CorruptedPDF%i" % numLayer)
    # plt.plot(bins[1:], layercdf, color="green", label="CDF%i" %numLayer)
    # plt.plot(bins_count[numLayer][1:], cdfs[numLayer], color="red", label="CorruptedCDF%i" %numLayer)
    # plt.xlim([-1.5, 1])
    # plt.legend()
    # plt.figtext(.8, .8, "Layer Size = %d"%len(layer))
    # #plt.show()
    # plt.savefig('./Results/RoundingSim/Layer%03d.png'%numLayer)
    # plt.clf()

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

def recurWeights(layers):
    global layerNum
    if np.isscalar(layers[0]):
        #testStats(layer, layerNum)
        checkdists(layers, layerNum)
        layerNum += 1
    else:
        for j in layers:
            recurWeights(j)

print(os.getcwd())
filename = "Model_MalwareCDNN_elu_weights.pickle"
f = open("./Networks/CypherBitCorrupted/%s"%filename, "rb")
weights = pickle.load(f)
weights = np.asarray(weights, dtype=object)
weights[-1] = weights[-1].flatten()


g = open("./elu/pdfs%s"%filename, "rb")
pdfs = pickle.load(g)
h = open("./elu/cdfs%s"%filename, "rb")
cdfs = pickle.load(h)
b = open("./elu/bins%s"%filename, "rb")
bins_count = pickle.load(b)

errorLayers = []
layerNum = 0

recurWeights(weights)