import pickle
import numpy as np

def plotDist(layer):
    global distsPDF, distsCDF
    # Getting data of the histogram
    count, bins = np.histogram(layer, bins=100)
    # Finding the PDF of the histogram using count values
    pdf = count / sum(count)
    # Calculate the CDF
    cdf = np.cumsum(pdf)
    # Add to final
    distsCDF.append(cdf)
    distsPDF.append(pdf)


def recurWeights(layer):
    if np.isscalar(layer[0]):
        plotDist(layer)
    else:
        for j in layer:
            recurWeights(j)


def initializeDists(network_weights):
    weights = np.asarray(network_weights, dtype=object)
    weights[-1] = weights[-1].flatten()
    global distsPDF, distsCDF
    recurWeights(weights)

    pickle.dump(distsPDF, open("./pdfsMNIST", "wb"))
    pickle.dump(distsCDF, open("./cdfsMNIST", "wb"))


distsPDF = []
distsCDF = []

# directory = r'./Networks/Original'
#
# for filename in os.listdir(directory):
#     f = open("./Networks/Original/%s"%filename, "rb")
#     weights = pickle.load(f)
#     weights = np.asarray(weights, dtype=object)
#     weights[-1] = weights[-1].flatten()
#
#     # Temp arrays for code clarity
#     # avg = []
#     # maxi = []
#     # mini = []
#     output = []
#     distsPDF = []
#     distsCDF = []
#     bins_count = []
#     totalNetwork = []
#
#     layerNum = 0
#     recurWeights(weights)
#     count, bins = np.histogram(totalNetwork, bins=100)
#     networkpdf = count / sum(count)
#     networkcdf = np.cumsum(networkpdf)
#     distsCDF.append(networkcdf)
#     distsPDF.append(networkpdf)
#     bins_count.append(bins)
#
#     # Converting temp arrays to final output
#     # for i in range(layerNum):
#     #     output.append([i, avg[i], mini[i], maxi[i]])
#
#     # pickle.dump(output, open("./stats%s"%filename, "wb"))
#     pickle.dump(distsPDF, open("./pdfs%s"%filename, "wb"))
#     pickle.dump(distsCDF, open("./cdfs%s"%filename, "wb"))
#     pickle.dump(bins_count, open("./bins%s"%filename, "wb"))
