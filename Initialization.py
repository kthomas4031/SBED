import pickle
import numpy as np
import matplotlib.pyplot as plt

# f = open("./Model_1_elu_weights.pickle", "rb")
f = open("./Networks/Model_MNIST_elu_weights.pickle", "rb")

weights = pickle.load(f)
weights = np.asarray(weights, dtype=object)

# Temp arrays for code clarity
avg = []
maxi = []
mini = []
output = []
distsPDF = []
distsCDF = []
bins_count = []

layerNum = 0

def plotDist(layer, numLayer):
    # Getting data of the histogram
    count, bins = np.histogram(layer, bins=10)

    # Finding the PDF of the histogram using count values
    pdf = count / sum(count)

    # Calculate the CDF
    cdf = np.cumsum(pdf)


    # Add to final
    distsCDF.append(cdf)
    distsPDF.append(pdf)
    bins_count.append(bins)

    # Plotting PDF and CDF
    # plt.plot(bins[1:], pdf, color="red", label="PDF%i" %numLayer)
    # plt.plot(bins[1:], cdf, label="CDF%i" %numLayer)
    # plt.legend()

def findStats(layer):
    # Initialize stats for current layer to ensure consistency
    denom = 0
    tempMin = 99
    tempMax = 0
    tempAvg = 0

    #print(layer)

    # Iterate through parameters in the layer and calculate stats
    for parameter in layer:
        denom += 1
        tempAvg += parameter

        if parameter > tempMax:
            tempMax = parameter
        elif parameter < tempMin:
            tempMin = parameter

    tempAvg = tempAvg / denom
    mini.append(tempMin)
    maxi.append(tempMax)
    avg.append(tempAvg)

for i in weights:
    if type(i[0]) is np.ndarray:
        for a in i:
            if type(a[0]) is np.ndarray:
                for b in a:
                    if type(b[0]) is np.ndarray:
                        for c in b:
                            findStats(c)
                            plotDist(c, layerNum)
                            layerNum += 1
                    else:
                        findStats(b)
                        plotDist(b, layerNum)
                        layerNum += 1
            else:
                findStats(a)
                plotDist(a,layerNum)
                layerNum += 1
    else:
        findStats(i)
        plotDist(i, layerNum)
        layerNum += 1

# Converting temp arrays to final output
for i in range(layerNum):
    output.append([i, avg[i], mini[i], maxi[i]])

pickle.dump(output, open("MNISTmodelStats.pickle", "wb"))
pickle.dump(distsPDF, open("MNISTlayerpdfs.pickle", "wb"))
pickle.dump(distsCDF, open("MNISTlayercdfs.pickle", "wb"))
pickle.dump(bins_count, open("MNISTbins.pickle", "wb"))
# pickle.dump(output, open("modelStats.pickle", "wb"))
# pickle.dump(distsPDF, open("layerpdfs.pickle", "wb"))
# pickle.dump(distsCDF, open("layercdfs.pickle", "wb"))
# pickle.dump(bins_count, open("bins.pickle", "wb"))
#plt.show()

#f = open("./modelStats.pickle", "rb")
#stats = pickle.load(f)
